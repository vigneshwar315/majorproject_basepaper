"""Deterministic validation of MCDM output: ids must exist and be feasible; claims are checked against the matrix."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CLAIM = re.compile(r"\b([zd]\d{1,2})\s*(?:=|:|is|of|equals|at)?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_METRIC_REF = re.compile(r"\b([zd]\d{1,2})\b", re.IGNORECASE)


@dataclass
class SelectionValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    claims_checked: int = 0
    claims_supported: int = 0
    unsupported_claims: list[str] = field(default_factory=list)
    selected_plan_claims_matching: int = 0

    @property
    def grounding_rate(self):
        return None if self.claims_checked == 0 else self.claims_supported / self.claims_checked

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "errors": self.errors, "warnings": self.warnings, "claims_checked": self.claims_checked,
            "claims_supported": self.claims_supported, "unsupported_claims": self.unsupported_claims,
            "grounding_rate": self.grounding_rate, "selected_plan_claims_matching": self.selected_plan_claims_matching,
        }


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.011, 0.005 * max(abs(a), abs(b)))


def check_grounding(justification: str, selected_id: str, candidates: list[dict], metric_ids: list[str]) -> SelectionValidation:
    """Compare numeric claims 'zK = value' in the justification with the real matrix (no LLM involved)."""
    v = SelectionValidation(True)
    by_id = {c["plan_id"]: c for c in candidates}
    for m in _METRIC_REF.finditer(justification):
        mid = m.group(1).lower()
        if mid not in metric_ids:
            v.warnings.append(f"justification mentions metric {mid} that is not in the evaluation matrix")
    for mid, num in _CLAIM.findall(justification):
        mid = mid.lower()
        if mid not in metric_ids:
            continue
        val = float(num)
        v.claims_checked += 1
        if any(_close(val, c["metrics"][mid]) for c in candidates):
            v.claims_supported += 1
            if selected_id in by_id and _close(val, by_id[selected_id]["metrics"][mid]):
                v.selected_plan_claims_matching += 1
        else:
            v.unsupported_claims.append(f"{mid} = {num}")
    return v


def validate_selection(data: dict, candidates: list[dict], metric_ids: list[str]) -> SelectionValidation:
    """``candidates`` are the dicts passed to the model (plan_id, metrics, feasible)."""
    errs: list[str] = []
    sel = data.get("selected_plan_id")
    just = data.get("justification")
    crit = data.get("criteria_used", [])
    by_id = {c["plan_id"]: c for c in candidates}
    if not isinstance(sel, str) or not sel:
        errs.append("selected_plan_id is missing or not a string")
    elif sel not in by_id:
        errs.append(f"selected_plan_id {sel!r} does not exist in the evaluation matrix (invented id)")
    elif not by_id[sel].get("feasible", False):
        errs.append(f"plan {sel!r} failed the deterministic feasibility check")
    if not isinstance(just, str) or not just.strip():
        errs.append("justification is missing or empty")
    if not isinstance(crit, list) or any(not isinstance(x, str) for x in crit):
        errs.append("criteria_used must be a list of metric ids")
    else:
        unknown = [x for x in crit if x not in metric_ids]
        if unknown:
            errs.append(f"criteria_used contains unknown metrics {unknown}")
    if errs:
        return SelectionValidation(False, errs)
    g = check_grounding(just, sel, candidates, metric_ids)
    return g
