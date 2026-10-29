"""Pure helper functions of the UI (no Streamlit import) so that they can be unit-tested."""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from paper_replication.config import DOCS_DIR, LOGS_DIR, RESULTS_DIR
from paper_replication.constraints import plan_route_results
from paper_replication.problem_instance import DistanceModel, validate_instance
from paper_replication.redact import redact
from paper_replication.schemas import (
    PICKUP,
    Instance,
    Location,
    Plan,
    Request,
    TimeWindow,
    Vehicle,
)

PRIORITY_LABELS = {0: "normal", 1: "high"}
PRIORITY_VALUES = {v: k for k, v in PRIORITY_LABELS.items()}

VEHICLE_COLUMNS = ["id", "capacity", "shift_duration_min"]
REQUEST_COLUMNS = [
    "id", "customer", "arrival_day", "arrival_seq", "pickup_x", "pickup_y", "pickup_zip", "delivery_x", "delivery_y", "delivery_zip",
    "demand", "priority", "pickup_tw_from", "pickup_tw_to", "delivery_tw_from", "delivery_tw_to", "service_time_min",
]


@dataclass
class FieldError:
    section: str  # "depot" | "vehicles" | "requests" | "horizon" | "travel"
    row: Optional[str]  # row label (vehicle/request id) or None
    column: Optional[str]
    message: str

    def __str__(self) -> str:
        where = self.section + (f" / {self.row}" if self.row else "") + (f" / {self.column}" if self.column else "")
        return f"{where}: {self.message}"


# ---- time helpers -------------------------------------------------------------------------------------------
def parse_hhmm(text) -> float:
    """'08:30' -> 510.0 (minutes since midnight). Raises ValueError with a readable message."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        raise ValueError("empty time")
    s = str(text).strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if not m:
        raise ValueError(f"'{s}' is not a time in HH:MM format")
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 47 or mi > 59:
        raise ValueError(f"'{s}' is not a valid time")
    return float(h * 60 + mi)


def fmt_hhmm(minutes: Optional[float]) -> str:
    if minutes is None or (isinstance(minutes, float) and pd.isna(minutes)):
        return ""
    total = int(round(minutes))
    return f"{total // 60:02d}:{total % 60:02d}"


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and not v.strip())


# ---- instance <-> editable frames -----------------------------------------------------------------------------
def instance_to_frames(inst: Instance) -> tuple[pd.DataFrame, pd.DataFrame]:
    veh = pd.DataFrame([{"id": v.id, "capacity": v.capacity, "shift_duration_min": v.shift_duration} for v in inst.vehicles], columns=VEHICLE_COLUMNS)
    rows = []
    for r in inst.requests:
        rows.append({
            "id": r.id, "customer": r.customer_id or "", "arrival_day": r.arrival_day, "arrival_seq": r.arrival_seq,
            "pickup_x": r.pickup.x, "pickup_y": r.pickup.y, "pickup_zip": r.pickup.zip,
            "delivery_x": r.delivery.x, "delivery_y": r.delivery.y, "delivery_zip": r.delivery.zip,
            "demand": r.demand, "priority": PRIORITY_LABELS.get(r.priority, str(r.priority)),
            "pickup_tw_from": fmt_hhmm(r.pickup_tw.earliest) if r.pickup_tw else "", "pickup_tw_to": fmt_hhmm(r.pickup_tw.latest) if r.pickup_tw else "",
            "delivery_tw_from": fmt_hhmm(r.delivery_tw.earliest) if r.delivery_tw else "", "delivery_tw_to": fmt_hhmm(r.delivery_tw.latest) if r.delivery_tw else "",
            "service_time_min": r.service_time,
        })
    return veh, pd.DataFrame(rows, columns=REQUEST_COLUMNS)


def scalar_settings(inst: Instance) -> dict:
    return {
        "name": inst.name, "depot_x": inst.depot.x, "depot_y": inst.depot.y, "depot_zip": inst.depot.zip,
        "horizon_days": inst.horizon_days, "current_day": inst.current_day, "shift_start": fmt_hhmm(inst.shift_start),
        "depot_close": fmt_hhmm(inst.depot_close), "minutes_per_unit": inst.minutes_per_unit, "max_lead_days": inst.max_lead_days,
        "synthetic": inst.synthetic, "description": inst.description,
    }


def _num(v, kind=float):
    if _blank(v):
        raise ValueError("required")
    try:
        x = kind(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{v}' is not a number") from None
    return x


def _int(v):
    x = _num(v, float)
    if x != int(x):
        raise ValueError(f"'{v}' must be a whole number")
    return int(x)


def parse_distance_matrix_csv(text: str) -> tuple[Optional[dict], list[str]]:
    """Parse a square CSV (first column = node ids, header = node ids) into {from: {to: value}}."""
    try:
        df = pd.read_csv(io.StringIO(text), index_col=0)
    except Exception as exc:  # noqa: BLE001
        return None, [f"could not read the CSV: {exc}"]
    errs = []
    out: dict[str, dict[str, float]] = {}
    for a in df.index.astype(str):
        out[a] = {}
        for b in df.columns.astype(str):
            if a == b:
                continue
            val = df.loc[a if a in df.index else df.index[list(df.index.astype(str)).index(a)], b]
            try:
                out[a][b] = float(val)
            except (TypeError, ValueError):
                errs.append(f"entry {a} -> {b} is not a number")
    return out, errs


def frames_to_instance(settings: dict, vehicles: pd.DataFrame, requests: pd.DataFrame, distance_matrix: Optional[dict] = None) -> tuple[Optional[Instance], list[FieldError]]:
    """Convert the editable frames into an ``Instance``. Returns field-level errors instead of raising."""
    errs: list[FieldError] = []
    veh_list: list[Vehicle] = []
    for i, row in vehicles.reset_index(drop=True).iterrows():
        label = str(row.get("id") or f"row {i + 1}")
        if _blank(row.get("id")):
            errs.append(FieldError("vehicles", f"row {i + 1}", "id", "id is required"))
            continue
        try:
            cap = _int(row.get("capacity"))
        except ValueError as e:
            errs.append(FieldError("vehicles", label, "capacity", str(e)))
            continue
        try:
            shift = _num(row.get("shift_duration_min"))
        except ValueError as e:
            errs.append(FieldError("vehicles", label, "shift_duration_min", str(e)))
            continue
        veh_list.append(Vehicle(str(row["id"]).strip(), cap, shift))
    req_list: list[Request] = []
    for i, row in requests.reset_index(drop=True).iterrows():
        rid = str(row.get("id") or "").strip()
        label = rid or f"row {i + 1}"
        if not rid:
            errs.append(FieldError("requests", label, "id", "id is required"))
            continue
        bad = False

        def get(col, conv):
            nonlocal bad
            try:
                return conv(row.get(col))
            except ValueError as e:
                errs.append(FieldError("requests", label, col, str(e)))
                bad = True
                return None

        arrival_day = get("arrival_day", _int)
        arrival_seq = get("arrival_seq", lambda v: _int(v) if not _blank(v) else i)
        px, py = get("pickup_x", _num), get("pickup_y", _num)
        dx, dy = get("delivery_x", _num), get("delivery_y", _num)
        demand = get("demand", _int)
        service = get("service_time_min", lambda v: 0.0 if _blank(v) else _num(v))
        pr = str(row.get("priority") or "normal").strip().lower()
        if pr not in PRIORITY_VALUES:
            errs.append(FieldError("requests", label, "priority", "priority must be 'normal' or 'high'"))
            bad = True

        def tw(prefix):
            nonlocal bad
            a, b = row.get(f"{prefix}_tw_from"), row.get(f"{prefix}_tw_to")
            if _blank(a) and _blank(b):
                return None
            if _blank(a) or _blank(b):
                errs.append(FieldError("requests", label, f"{prefix}_tw_from", "give both the start and the end of the time window (or neither)"))
                bad = True
                return None
            try:
                return TimeWindow(parse_hhmm(a), parse_hhmm(b))
            except ValueError as e:
                errs.append(FieldError("requests", label, f"{prefix}_tw_from", str(e)))
                bad = True
                return None

        ptw, dtw = tw("pickup"), tw("delivery")
        if bad:
            continue
        req_list.append(Request(
            id=rid, arrival_day=arrival_day, arrival_seq=arrival_seq, pickup=Location(px, py, str(row.get("pickup_zip") or "").strip()),
            delivery=Location(dx, dy, str(row.get("delivery_zip") or "").strip()), demand=demand, priority=PRIORITY_VALUES[pr],
            pickup_tw=ptw, delivery_tw=dtw, service_time=service, customer_id=(str(row.get("customer")).strip() or None) if not _blank(row.get("customer")) else None,
        ))
    try:
        depot = Location(_num(settings["depot_x"]), _num(settings["depot_y"]), str(settings.get("depot_zip") or ""))
    except ValueError as e:
        errs.append(FieldError("depot", None, "coordinates", str(e)))
        depot = Location(0.0, 0.0, "")
    horizon: dict = {}
    for key, conv in (("horizon_days", _int), ("current_day", _int), ("max_lead_days", _int), ("minutes_per_unit", _num)):
        try:
            horizon[key] = conv(settings[key])
        except (ValueError, KeyError) as e:
            errs.append(FieldError("horizon", None, key, str(e)))
    for key in ("shift_start", "depot_close"):
        try:
            horizon[key] = parse_hhmm(settings[key])
        except (ValueError, KeyError) as e:
            errs.append(FieldError("horizon", None, key, str(e)))
    if errs:
        return None, errs
    inst = Instance(
        name=str(settings.get("name") or "custom_instance"), depot=depot, vehicles=veh_list, requests=req_list,
        horizon_days=horizon["horizon_days"], current_day=horizon["current_day"], shift_start=horizon["shift_start"], depot_close=horizon["depot_close"],
        minutes_per_unit=horizon["minutes_per_unit"], max_lead_days=horizon["max_lead_days"], distance_matrix=distance_matrix,
        synthetic=bool(settings.get("synthetic", False)), description=str(settings.get("description", "")),
    )
    for msg in validate_instance(inst):
        errs.append(_classify_error(msg))
    return (None if errs else inst), errs


def _classify_error(msg: str) -> FieldError:
    m = re.match(r"(request|vehicle) (\S+?):? (.*)", msg)
    if m:
        kind, ident, rest = m.groups()
        col = None
        for c in ("capacity", "shift_duration", "demand", "priority", "arrival_day", "pickup_tw", "delivery_tw", "service_time"):
            if c in rest:
                col = c
        return FieldError("requests" if kind == "request" else "vehicles", ident.rstrip(":"), col, rest.lstrip(": "))
    if msg.startswith("duplicate request"):
        return FieldError("requests", None, "id", msg)
    if msg.startswith("duplicate vehicle"):
        return FieldError("vehicles", None, "id", msg)
    if msg.startswith("distance_matrix"):
        return FieldError("travel", None, None, msg)
    if "vehicle" in msg:
        return FieldError("vehicles", None, None, msg)
    return FieldError("horizon", None, None, msg)


def errors_by_section(errors: list[FieldError]) -> dict[str, list[FieldError]]:
    out: dict[str, list[FieldError]] = {}
    for e in errors:
        out.setdefault(e.section, []).append(e)
    return out


def node_ids_for_template(inst: Instance) -> list[str]:
    return list(inst.node_locations())


def distance_matrix_template_csv(inst: Instance) -> str:
    dm = DistanceModel(inst)
    nodes = dm.nodes
    df = pd.DataFrame([[round(dm.dist(a, b), 4) for b in nodes] for a in nodes], index=nodes, columns=nodes)
    return df.to_csv()


# ---- plan tables ----------------------------------------------------------------------------------------------
def route_table(inst: Instance, plan: Plan) -> pd.DataFrame:
    """Detailed stop-by-stop table for every route of a plan (timings from the deterministic validator)."""
    reqs = inst.request_map()
    results = plan_route_results(inst, plan)
    rows = []
    for rt in sorted(plan.routes, key=lambda r: (r.day, r.vehicle_id)):
        res = results[(rt.day, rt.vehicle_id)]
        for k, (st, tm) in enumerate(zip(rt.stops, res.stops), start=1):
            r = reqs[st.request_id]
            loc = r.pickup if st.kind == PICKUP else r.delivery
            rows.append({
                "day": rt.day, "vehicle": rt.vehicle_id, "stop": k, "request": st.request_id, "type": "pickup" if st.kind == PICKUP else "delivery",
                "zip": loc.zip, "x": loc.x, "y": loc.y, "arrival": fmt_hhmm(tm.arrival), "service start": fmt_hhmm(tm.start), "waiting (min)": round(tm.wait, 1),
                "load after": tm.load_after, "priority": PRIORITY_LABELS[r.priority],
            })
    cols = ["day", "vehicle", "stop", "request", "type", "zip", "x", "y", "arrival", "service start", "waiting (min)", "load after", "priority"]
    return pd.DataFrame(rows, columns=cols)


def route_summary_table(inst: Instance, plan: Plan) -> pd.DataFrame:
    results = plan_route_results(inst, plan)
    rows = []
    for rt in sorted(plan.routes, key=lambda r: (r.day, r.vehicle_id)):
        res = results[(rt.day, rt.vehicle_id)]
        rows.append({"day": rt.day, "vehicle": rt.vehicle_id, "requests": len(rt.stops) // 2, "departs": fmt_hhmm(res.depart), "returns": fmt_hhmm(res.end),
                     "duration (min)": round(res.duration, 1), "distance": round(res.distance, 1), "waiting (min)": round(res.waiting, 1),
                     "sequence": " > ".join(f"{s.request_id}{s.kind}" for s in rt.stops), "feasible": res.feasible})
    return pd.DataFrame(rows, columns=["day", "vehicle", "requests", "departs", "returns", "duration (min)", "distance", "waiting (min)", "sequence", "feasible"])


def decisions_table(plan: Plan) -> pd.DataFrame:
    rows = [{"request": d.request_id, "decision": d.status, "day": d.day, "vehicle": d.vehicle_id, "reason": d.reason} for d in plan.decisions.values()]
    return pd.DataFrame(rows, columns=["request", "decision", "day", "vehicle", "reason"])


def comparison_table(result_plans: list[Plan], matrix: dict[str, dict[str, float]], metric_ids: list[str], selected: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for p in result_plans:
        row = {"plan": p.plan_id, "feasible": bool(p.feasible), "accepted": len(p.accepted_ids()), "rejected": len(p.rejected_ids()), "routes": len(p.routes),
               "policy": p.policy.label}
        vals = matrix.get(p.plan_id, {})
        for m in metric_ids:
            row[m] = vals.get(m)
        row["selected"] = "yes" if p.plan_id == selected else ""
        rows.append(row)
    return pd.DataFrame(rows)


def prompt_call_estimate(exp_id: str) -> int:
    """Rough number of provider calls an experiment makes (shown before running against a real API)."""
    return {"C1": 90 * 2, "C2": 30 * 3 * 3, "C3": 270 * 2, "C4": 30 * 3 * 3, "C5": 9 * 2 + 6 * 3, "C6": 30}.get(exp_id, 0)


# ---- documents and logs ---------------------------------------------------------------------------------------
DOCS = {
    "Source audit": DOCS_DIR / "source_audit.md",
    "Methodology specification": DOCS_DIR / "paper_methodology.md",
    "Traceability matrix": DOCS_DIR / "requirements_traceability.csv",
    "Uncertainties and deviations": DOCS_DIR / "uncertainties_and_deviations.md",
    "Implementation plan": DOCS_DIR / "implementation_plan.md",
    "Experiment protocol": DOCS_DIR / "experiment_protocol.md",
    "Execution log": DOCS_DIR / "execution_log.md",
    "Repair log": LOGS_DIR / "repair_log.md",
    "Reproduction report": DOCS_DIR / "reproduction_report.md",
}


def read_doc(path: Path) -> Optional[str]:
    try:
        return redact(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def file_stamp(path: Path) -> str:
    try:
        import datetime as dt

        return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except OSError:
        return "not found"


def test_status_summary() -> Optional[dict]:
    """Real test status written by scripts/run_tests.py (None when tests were never run through that script)."""
    p = LOGS_DIR / "test_status.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def test_totals(ts: dict) -> dict:
    """Totals over the suites of logs/test_status.json (live tests that were NOT RUN contribute nothing)."""
    out = {"passed": 0, "failed": 0, "skipped": 0, "tests": 0}
    for suite in (ts.get("suites") or {}).values():
        for k in out:
            out[k] += int(suite.get(k, 0) or 0)
    return out


def list_runs(results_dir: Optional[Path] = None) -> list[dict]:
    base = (results_dir or RESULTS_DIR) / "runs"
    out = []
    if base.exists():
        for d in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
            try:
                out.append(json.loads((d / "record.json").read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    return out
