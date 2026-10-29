"""Deterministic route timing and plan validation.

Timing recursion (paper p.149784, IV-B1):  a_j = d_i + t_ij,  s_j = max(e_j, a_j),  w_j = max(0, e_j - a_j),
d_j = s_j (zero service duration; ``Request.service_time`` extends this and defaults to 0).

This module is the ONLY place that sets ``Plan.feasible``. Hard constraints are never delegated to an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .problem_instance import DistanceModel
from .schemas import DEPOT_NODE, PICKUP, Instance, Plan, PolicyParams, Request, Stop, Vehicle

EPS = 1e-9


@dataclass
class StopTiming:
    request_id: str
    kind: str
    arrival: float
    start: float
    depart: float
    wait: float
    load_after: int


@dataclass
class RouteResult:
    feasible: bool = True
    violations: list[str] = field(default_factory=list)
    depart: float = 0.0
    end: float = 0.0
    duration: float = 0.0
    distance: float = 0.0
    waiting: float = 0.0
    stops: list[StopTiming] = field(default_factory=list)
    max_load: int = 0


def _tw(req: Request, kind: str):
    return req.pickup_tw if kind == PICKUP else req.delivery_tw


def evaluate_route(
    inst: Instance,
    dm: DistanceModel,
    reqs: dict[str, Request],
    vehicle: Vehicle,
    stops: list[Stop],
    *,
    policy: Optional[PolicyParams] = None,
    stop_at_first: bool = False,
) -> RouteResult:
    """Simulate one vehicle-day route. ``policy`` adds *policy* checks (prefixed ``POLICY:``)."""
    res = RouteResult()
    viol = res.violations

    def fail(msg: str) -> bool:
        viol.append(msg)
        res.feasible = False
        return stop_at_first

    if not stops:
        res.depart = res.end = inst.shift_start
        return res
    mpu = inst.minutes_per_unit
    first = stops[0]
    freq = reqs.get(first.request_id)
    if freq is None:
        fail(f"unknown request {first.request_id!r}")
        return res
    tw0 = _tw(freq, first.kind)
    e0 = tw0.earliest if tw0 else 0.0
    t0 = dm.dist(DEPOT_NODE, first.node) * mpu
    depart = max(inst.shift_start, e0 - t0)
    res.depart = depart
    time = depart
    prev = DEPOT_NODE
    load = 0
    normal_load = 0
    picked: set[str] = set()
    delivered: set[str] = set()
    normal_cap = None
    shift_cap = vehicle.shift_duration
    if policy is not None:
        if policy.reserve_fraction > 0:
            normal_cap = vehicle.capacity * (1.0 - policy.reserve_fraction)
        if policy.shift_limit is not None:
            shift_cap = min(shift_cap, policy.shift_limit)
    for st in stops:
        r = reqs.get(st.request_id)
        if r is None:
            if fail(f"unknown request {st.request_id!r}"):
                return res
            continue
        d_leg = dm.dist(prev, st.node)
        res.distance += d_leg
        arrival = time + d_leg * mpu
        tw = _tw(r, st.kind)
        e = tw.earliest if tw else 0.0
        l = tw.latest if tw else float("inf")
        start = max(arrival, e)
        wait = max(0.0, e - arrival)
        if start > l + EPS and fail(f"time window violated at {st.node}: service start {start:.2f} > latest {l:.2f}"):
            return res
        depart_i = start + r.service_time  # paper: d_j = s_j (service_time defaults to 0)
        if st.kind == PICKUP:
            if st.request_id in picked and fail(f"request {st.request_id} picked up twice"):
                return res
            picked.add(st.request_id)
            load += r.demand
            if r.priority == 0:
                normal_load += r.demand
            if load > vehicle.capacity and fail(f"capacity exceeded at {st.node}: load {load} > {vehicle.capacity}"):
                return res
            if normal_cap is not None and r.priority == 0 and normal_load > normal_cap + EPS and fail("POLICY: reserved capacity for high-priority requests"):
                return res
        else:
            if st.request_id not in picked and fail(f"delivery before pickup for request {st.request_id}"):
                return res
            if st.request_id in delivered and fail(f"request {st.request_id} delivered twice"):
                return res
            delivered.add(st.request_id)
            load -= r.demand
            if r.priority == 0:
                normal_load -= r.demand
        if policy is not None and policy.avoid_zip:
            loc = r.pickup if st.kind == PICKUP else r.delivery
            if loc.zip == policy.avoid_zip and policy.peak_start - EPS <= start < policy.peak_end and fail(
                f"POLICY: service in ZIP {policy.avoid_zip} during peak window at {st.node}"
            ):
                return res
        res.waiting += wait
        res.max_load = max(res.max_load, load)
        res.stops.append(StopTiming(st.request_id, st.kind, arrival, start, depart_i, wait, load))
        time = depart_i
        prev = st.node
    for rid in picked - delivered:
        if fail(f"request {rid} picked up but never delivered"):
            return res
    back = dm.dist(prev, DEPOT_NODE)
    res.distance += back
    res.end = time + back * mpu
    res.duration = res.end - depart
    if res.end > inst.depot_close + EPS and fail(f"route ends at {res.end:.2f} after depot close {inst.depot_close:.2f}"):
        return res
    if res.duration > shift_cap + EPS and fail(
        ("POLICY: " if shift_cap < vehicle.shift_duration - EPS else "") + f"route duration {res.duration:.2f} exceeds shift {shift_cap:.2f}"
    ):
        return res
    return res


def plan_route_results(inst: Instance, plan: Plan, dm: Optional[DistanceModel] = None) -> dict[tuple[int, str], RouteResult]:
    dm = dm or DistanceModel(inst)
    reqs = inst.request_map()
    vmap = inst.vehicle_map()
    out = {}
    for r in plan.routes:
        v = vmap.get(r.vehicle_id)
        if v is None:
            out[(r.day, r.vehicle_id)] = RouteResult(False, [f"unknown vehicle {r.vehicle_id!r}"])
            continue
        out[(r.day, r.vehicle_id)] = evaluate_route(inst, dm, reqs, v, r.stops)
    return out


def validate_plan(inst: Instance, plan: Plan, dm: Optional[DistanceModel] = None) -> list[str]:
    """Return every hard-constraint violation of ``plan`` (empty list = feasible)."""
    viol: list[str] = []
    dm = dm or DistanceModel(inst)
    reqs = inst.request_map()
    vmap = inst.vehicle_map()
    received = {r.id for r in inst.received_requests()}
    seen_routes: set[tuple[int, str]] = set()
    placed: dict[str, list[tuple[int, str, str]]] = {}
    for rt in plan.routes:
        key = (rt.day, rt.vehicle_id)
        if key in seen_routes:
            viol.append(f"duplicate route for vehicle {rt.vehicle_id} on day {rt.day}")
        seen_routes.add(key)
        if rt.vehicle_id not in vmap:
            viol.append(f"unknown vehicle {rt.vehicle_id!r}")
            continue
        if not (0 <= rt.day < inst.horizon_days):
            viol.append(f"route day {rt.day} outside horizon [0,{inst.horizon_days})")
        for st in rt.stops:
            placed.setdefault(st.request_id, []).append((rt.day, rt.vehicle_id, st.kind))
        res = evaluate_route(inst, dm, reqs, vmap[rt.vehicle_id], rt.stops)
        for v in res.violations:
            viol.append(f"route ({rt.vehicle_id}, day {rt.day}): {v}")
    for rid, places in placed.items():
        r = reqs.get(rid)
        if r is None:
            viol.append(f"unknown request {rid!r} in plan")
            continue
        kinds = sorted(k for _, _, k in places)
        if kinds != ["D", "P"]:
            viol.append(f"request {rid} must appear exactly once as pickup and once as delivery (got {kinds})")
        if len({(d, v) for d, v, _ in places}) != 1:
            viol.append(f"request {rid} pickup and delivery are on different routes")
        day = places[0][0]
        if not (r.arrival_day <= day <= r.arrival_day + inst.max_lead_days):
            viol.append(f"request {rid} scheduled on day {day} outside [{r.arrival_day}, {r.arrival_day + inst.max_lead_days}]")
    acc = {rid for rid, d in plan.decisions.items() if d.status == "accepted"}
    rej = {rid for rid, d in plan.decisions.items() if d.status == "rejected"}
    for rid, d in plan.decisions.items():
        if d.status not in ("accepted", "rejected"):
            viol.append(f"request {rid}: invalid status {d.status!r}")
    if acc != set(placed):
        viol.append(f"accepted set {sorted(acc ^ set(placed))} inconsistent with routes")
    if acc & rej:
        viol.append("requests both accepted and rejected")
    if (acc | rej) != received:
        viol.append(f"decisions must cover exactly the received requests (diff {sorted((acc | rej) ^ received)})")
    return viol


def finalize_plan(inst: Instance, plan: Plan, dm: Optional[DistanceModel] = None) -> Plan:
    """Run the validator and record the outcome on the plan. The only writer of ``Plan.feasible``."""
    plan.violations = validate_plan(inst, plan, dm)
    plan.feasible = not plan.violations
    return plan
