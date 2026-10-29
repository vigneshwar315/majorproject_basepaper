"""Instance validation, distance model and a seeded synthetic instance generator.

The paper gives no dataset (NOT AVAILABLE) and no generator (NOT SPECIFIED). The generator below is an
implementation decision; every instance it produces is flagged ``synthetic=True``.
"""

from __future__ import annotations

import math

import numpy as np

from .schemas import (
    DELIVERY,
    DEPOT_NODE,
    HIGH,
    NORMAL,
    PICKUP,
    Instance,
    InstanceValidationError,
    Location,
    Request,
    TimeWindow,
    Vehicle,
    is_finite_number,
)


def validate_instance(inst: Instance) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errs: list[str] = []
    if not isinstance(inst.horizon_days, int) or inst.horizon_days < 1:
        errs.append("horizon_days must be an integer >= 1")
    if not isinstance(inst.max_lead_days, int) or inst.max_lead_days < 0:
        errs.append("max_lead_days must be an integer >= 0")
    if not is_finite_number(inst.minutes_per_unit) or inst.minutes_per_unit <= 0:
        errs.append("minutes_per_unit must be a positive number")
    if not (is_finite_number(inst.shift_start) and is_finite_number(inst.depot_close)) or inst.shift_start >= inst.depot_close:
        errs.append("shift_start must be earlier than depot_close")
    if not (is_finite_number(inst.depot.x) and is_finite_number(inst.depot.y)):
        errs.append("depot coordinates must be finite numbers")
    if not inst.vehicles:
        errs.append("at least one vehicle is required")
    seen_v: set[str] = set()
    for v in inst.vehicles:
        if v.id in seen_v:
            errs.append(f"duplicate vehicle id {v.id!r}")
        seen_v.add(v.id)
        if not isinstance(v.capacity, int) or v.capacity <= 0:
            errs.append(f"vehicle {v.id}: capacity must be a positive integer")
        if not is_finite_number(v.shift_duration) or v.shift_duration <= 0:
            errs.append(f"vehicle {v.id}: shift_duration must be positive")
    seen_r: set[str] = set()
    for r in inst.requests:
        p = f"request {r.id}"
        if r.id in seen_r:
            errs.append(f"duplicate request id {r.id!r}")
        seen_r.add(r.id)
        if ":" in r.id or not r.id:
            errs.append(f"{p}: id must be non-empty and must not contain ':'")
        if not isinstance(r.demand, int) or r.demand <= 0:
            errs.append(f"{p}: demand must be a positive integer")
        if r.priority not in (NORMAL, HIGH):
            errs.append(f"{p}: priority must be 0 (normal) or 1 (high)")
        if not isinstance(r.arrival_day, int) or not (0 <= r.arrival_day < max(inst.horizon_days, 1)):
            errs.append(f"{p}: arrival_day must be in [0, horizon_days)")
        elif inst.current_day is not None and r.arrival_day > inst.current_day:
            errs.append(f"{p}: arrives after current_day ({r.arrival_day} > {inst.current_day})")
        for nm, loc in (("pickup", r.pickup), ("delivery", r.delivery)):
            if not (is_finite_number(loc.x) and is_finite_number(loc.y)):
                errs.append(f"{p}: {nm} coordinates must be finite numbers")
        for nm, tw in (("pickup_tw", r.pickup_tw), ("delivery_tw", r.delivery_tw)):
            if tw is not None:
                if not (is_finite_number(tw.earliest) and is_finite_number(tw.latest)):
                    errs.append(f"{p}: {nm} bounds must be finite numbers")
                elif tw.earliest > tw.latest:
                    errs.append(f"{p}: {nm} earliest must be <= latest")
        if not is_finite_number(r.service_time) or r.service_time < 0:
            errs.append(f"{p}: service_time must be >= 0")
    if inst.current_day is None or not (0 <= inst.current_day < max(inst.horizon_days, 1)):
        errs.append("current_day must be in [0, horizon_days)")
    if inst.distance_matrix is not None:
        errs.extend(_validate_matrix(inst))
    return errs


def _validate_matrix(inst: Instance) -> list[str]:
    errs = []
    nodes = list(inst.node_locations())
    dm = inst.distance_matrix or {}
    missing = 0
    for a in nodes:
        row = dm.get(a)
        if row is None:
            errs.append(f"distance_matrix: missing row for node {a!r}")
            continue
        for b in nodes:
            if a == b:
                continue
            val = row.get(b)
            if val is None:
                missing += 1
                if missing <= 5:
                    errs.append(f"distance_matrix: missing entry {a!r} -> {b!r}")
            elif not is_finite_number(val) or val < 0:
                errs.append(f"distance_matrix: invalid entry {a!r} -> {b!r}: {val!r}")
    if missing > 5:
        errs.append(f"distance_matrix: {missing} entries missing in total")
    return errs


def ensure_valid(inst: Instance) -> Instance:
    errs = validate_instance(inst)
    if errs:
        raise InstanceValidationError(errs)
    return inst


class DistanceModel:
    """Dense node-to-node distance matrix with fast index lookup."""

    def __init__(self, inst: Instance):
        self.nodes = list(inst.node_locations())
        self.index = {n: i for i, n in enumerate(self.nodes)}
        locs = inst.node_locations()
        n = len(self.nodes)
        mat = np.zeros((n, n), dtype=float)
        if inst.distance_matrix is None:
            xy = np.array([[locs[k].x, locs[k].y] for k in self.nodes], dtype=float)
            diff = xy[:, None, :] - xy[None, :, :]
            mat = np.sqrt((diff**2).sum(axis=2))
        else:
            for a in self.nodes:
                for b in self.nodes:
                    if a != b:
                        mat[self.index[a], self.index[b]] = float(inst.distance_matrix[a][b])
        self.matrix = mat
        self.minutes_per_unit = inst.minutes_per_unit

    def dist(self, a: str, b: str) -> float:
        return float(self.matrix[self.index[a], self.index[b]])

    def travel(self, a: str, b: str) -> float:
        return self.dist(a, b) * self.minutes_per_unit


# ---------------------------------------------------------------------------------------------
# Synthetic generator (IMPLEMENTATION DECISION; the paper provides no dataset)
# ---------------------------------------------------------------------------------------------
def _zip_for(x: float, y: float, size: float, zones: int) -> str:
    zx = min(int(x / (size / zones)), zones - 1)
    zy = min(int(y / (size / zones)), zones - 1)
    return f"{10000 + zy * zones + zx:05d}"


def generate_instance(
    seed: int = 20251028,
    n_requests: int = 60,
    n_vehicles: int = 3,
    capacity: int = 6,
    horizon_days: int = 4,
    current_day: int = 2,
    size: float = 120.0,
    tw_fraction: float = 0.4,
    tight_fraction: float = 0.35,
    high_priority_fraction: float = 0.25,
    n_customers: int | None = None,
    zones: int = 3,
    shift_duration: float = 480.0,
    name: str | None = None,
) -> Instance:
    """Seeded synthetic multi-period dynamic VRPPD instance.

    Requests arrive on days ``0..current_day`` (roughly evenly). Distances are Euclidean on a
    ``size x size`` plane (1 unit = 1 minute of driving by default).
    """
    if n_requests < 0 or n_vehicles < 1:
        raise InstanceValidationError(["n_requests must be >= 0 and n_vehicles >= 1"])
    rng = np.random.default_rng(seed)
    depot = Location(size / 2, size / 2, _zip_for(size / 2, size / 2, size, zones))
    vehicles = [Vehicle(f"V{i + 1}", capacity, shift_duration) for i in range(n_vehicles)]
    n_cust = n_customers or max(1, int(n_requests * 0.6))
    requests: list[Request] = []
    arrival_days = np.sort(rng.integers(0, current_day + 1, size=n_requests)) if n_requests else np.array([], dtype=int)
    seq_in_day: dict[int, int] = {}
    for i in range(n_requests):
        day = int(arrival_days[i])
        seq = seq_in_day.get(day, 0)
        seq_in_day[day] = seq + 1
        px, py = rng.uniform(0, size, 2)
        dx, dy = rng.uniform(0, size, 2)
        pickup = Location(round(float(px), 2), round(float(py), 2), _zip_for(px, py, size, zones))
        delivery = Location(round(float(dx), 2), round(float(dy), 2), _zip_for(dx, dy, size, zones))
        priority = HIGH if rng.random() < high_priority_fraction else NORMAL
        demand = int(rng.integers(1, 4))
        ptw = dtw = None
        if rng.random() < tw_fraction:
            tight = rng.random() < tight_fraction
            width = float(rng.choice([30, 45, 60])) if tight else float(rng.choice([120, 180, 240]))
            start = float(rng.integers(8 * 60 + 30, 15 * 60, endpoint=True))
            ptw = TimeWindow(start, start + width)
            if rng.random() < 0.5:
                dstart = start + float(rng.integers(30, 120))
                dtw = TimeWindow(dstart, dstart + width)
        requests.append(
            Request(
                id=f"r{i + 1}",
                arrival_day=day,
                arrival_seq=seq,
                pickup=pickup,
                delivery=delivery,
                demand=demand,
                priority=priority,
                pickup_tw=ptw,
                delivery_tw=dtw,
                customer_id=f"c{int(rng.integers(1, n_cust + 1))}",
            )
        )
    inst = Instance(
        name=name or f"synthetic_seed{seed}_n{n_requests}",
        depot=depot,
        vehicles=vehicles,
        requests=requests,
        horizon_days=horizon_days,
        current_day=current_day,
        synthetic=True,
        description="SYNTHETIC instance generated by paper_replication.problem_instance.generate_instance "
        "(the base paper provides no dataset).",
    )
    return ensure_valid(inst)


def tiny_instance() -> Instance:
    """Small hand-verifiable instance: depot at origin, 2 vehicles, 3 requests, no randomness."""
    depot = Location(0, 0, "00100")
    vehicles = [Vehicle("V1", 3, 480.0), Vehicle("V2", 3, 480.0)]
    reqs = [
        Request("a", 0, 0, Location(3, 4, "00100"), Location(6, 8, "00200"), demand=1, priority=NORMAL),
        Request("b", 0, 1, Location(0, 5, "00100"), Location(0, 10, "00200"), demand=2, priority=HIGH),
        Request("c", 1, 0, Location(10, 0, "00300"), Location(10, 5, "00300"), demand=1, priority=NORMAL,
                pickup_tw=TimeWindow(600, 660)),
    ]
    return ensure_valid(Instance("tiny", depot, vehicles, reqs, horizon_days=3, current_day=1, shift_start=480.0,
                                 depot_close=1200.0, max_lead_days=1, synthetic=False,
                                 description="Hand-built example for unit tests and documentation."))


def euclid(a: Location, b: Location) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


__all__ = [
    "DELIVERY", "DEPOT_NODE", "PICKUP", "DistanceModel", "ensure_valid", "euclid", "generate_instance",
    "tiny_instance", "validate_instance",
]
