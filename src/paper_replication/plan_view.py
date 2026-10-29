"""JSON-serialisable *plan view* consumed by metrics (reference implementations and the metric DSL).

A view contains plain records so that DMG examples can be stored as JSON and evaluated without the routing code.
"""

from __future__ import annotations

from typing import Optional

from .constraints import plan_route_results
from .problem_instance import DistanceModel
from .schemas import PICKUP, Instance, Plan

DEFAULT_PEAK = (690.0, 810.0)  # 11:30-13:30 (paper Table 4, prompt 5)
LATE_RETURN = 1080.0  # 18:00 (paper Table 4, prompt 8)
LONG_ROUTE_MIN = 420.0  # 7 h  ("long routes", prompt 9; threshold is our choice)
TIGHT_TW_WIDTH = 60.0  # minutes (prompt 3; threshold is our choice)


def default_target_zip(inst: Instance, plan: Plan) -> str:
    """ZIP with the most accepted visits (used for z8 when the user does not pick one)."""
    reqs = inst.request_map()
    counts: dict[str, int] = {}
    for rt in plan.routes:
        for st in rt.stops:
            r = reqs[st.request_id]
            z = (r.pickup if st.kind == PICKUP else r.delivery).zip
            counts[z] = counts.get(z, 0) + 1
    if not counts:
        zs = inst.zips()
        return zs[0] if zs else ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def build_plan_view(
    inst: Instance,
    plan: Plan,
    target_zip: Optional[str] = None,
    peak: tuple[float, float] = DEFAULT_PEAK,
    dm: Optional[DistanceModel] = None,
) -> dict:
    dm = dm or DistanceModel(inst)
    reqs = inst.request_map()
    results = plan_route_results(inst, plan, dm)
    today = inst.current_day
    tz = target_zip if target_zip is not None else default_target_zip(inst, plan)
    ctx = {
        "current_day": today,
        "tomorrow": today + 1,
        "num_vehicles": len(inst.vehicles),
        "num_requests_received": len(inst.received_requests()),
        "peak_start": peak[0],
        "peak_end": peak[1],
        "target_zip": tz,
        "late_return_time": LATE_RETURN,
        "long_route_minutes": LONG_ROUTE_MIN,
        "tight_window_width": TIGHT_TW_WIDTH,
    }
    long_days: dict[tuple[str, int], bool] = {}
    for rt in plan.routes:
        res = results[(rt.day, rt.vehicle_id)]
        long_days[(rt.vehicle_id, rt.day)] = res.duration >= LONG_ROUTE_MIN
    routes, visits, requests = [], [], []
    drivers_per_customer: dict[str, set] = {}
    for rid in plan.accepted_ids():
        d = plan.decisions[rid]
        drivers_per_customer.setdefault(reqs[rid].customer, set()).add(d.vehicle_id)
    for rt in sorted(plan.routes, key=lambda r: (r.day, r.vehicle_id)):
        res = results[(rt.day, rt.vehicle_id)]
        is_long = res.duration >= LONG_ROUTE_MIN
        routes.append(
            {
                "vehicle_id": rt.vehicle_id,
                "day": rt.day,
                "distance": res.distance,
                "duration": res.duration,
                "waiting": res.waiting,
                "num_stops": len(rt.stops),
                "num_requests": len(rt.stops) // 2,
                "start_time": res.depart,
                "end_time": res.end,
                "max_load": res.max_load,
                "is_today": rt.day == today,
                "is_tomorrow": rt.day == today + 1,
                "ends_late": res.end > LATE_RETURN,
                "is_long": is_long,
                "long_next_day": bool(is_long and long_days.get((rt.vehicle_id, rt.day + 1), False)),
            }
        )
        for st, tm in zip(rt.stops, res.stops):
            r = reqs[st.request_id]
            loc = r.pickup if st.kind == PICKUP else r.delivery
            visits.append(
                {
                    "request_id": st.request_id,
                    "kind": st.kind,
                    "zip": loc.zip,
                    "day": rt.day,
                    "vehicle_id": rt.vehicle_id,
                    "arrival": tm.arrival,
                    "service_start": tm.start,
                    "wait": tm.wait,
                    "is_today": rt.day == today,
                    "in_peak": peak[0] <= tm.start < peak[1],
                    "in_target_zip": loc.zip == tz,
                }
            )
    for r in inst.received_requests():
        d = plan.decisions.get(r.id)
        acc = bool(d and d.status == "accepted")
        mw = r.min_tw_width()
        sday = d.day if acc else -1
        requests.append(
            {
                "id": r.id,
                "customer": r.customer,
                "priority": r.priority,
                "is_priority": r.priority == 1,
                "accepted": acc,
                "scheduled_day": sday,
                "arrival_day": r.arrival_day,
                "arrived_today": r.arrival_day == today,
                "scheduled_today": acc and sday == today,
                "scheduled_future": acc and sday > today,
                "has_time_window": mw is not None,
                "tight_window": mw is not None and mw <= TIGHT_TW_WIDTH,
                "vehicle_id": d.vehicle_id if acc else "",
                "customer_num_drivers": len(drivers_per_customer.get(r.customer, ())) if acc else 0,
            }
        )
    return {"context": ctx, "routes": routes, "requests": requests, "visits": visits}
