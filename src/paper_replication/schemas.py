"""Validated data structures for the multi-period dynamic VRP with pickups and deliveries.

Source: base paper Sec. IV-A (p.149782). Fields not fixed by the paper (geometry, capacities, horizon...)
are implementation decisions documented in docs/uncertainties_and_deviations.md.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

DEPOT_NODE = "depot"
PICKUP = "P"
DELIVERY = "D"
NORMAL, HIGH = 0, 1


class InstanceValidationError(ValueError):
    """Raised when an instance is invalid; ``errors`` lists every problem found."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors) if self.errors else "invalid instance")


@dataclass(frozen=True)
class TimeWindow:
    earliest: float
    latest: float

    @property
    def width(self) -> float:
        return self.latest - self.earliest


@dataclass(frozen=True)
class Location:
    x: float
    y: float
    zip: str = ""


@dataclass(frozen=True)
class Vehicle:
    id: str
    capacity: int
    shift_duration: float = 480.0  # minutes (paper IV-C mentions 8 h shifts)


@dataclass(frozen=True)
class Request:
    id: str
    arrival_day: int
    arrival_seq: int
    pickup: Location
    delivery: Location
    demand: int = 1
    priority: int = NORMAL
    pickup_tw: Optional[TimeWindow] = None
    delivery_tw: Optional[TimeWindow] = None
    service_time: float = 0.0  # paper formula d_j = s_j  => zero service duration
    customer_id: Optional[str] = None

    @property
    def customer(self) -> str:
        return self.customer_id or self.id

    def min_tw_width(self) -> Optional[float]:
        widths = [tw.width for tw in (self.pickup_tw, self.delivery_tw) if tw is not None]
        return min(widths) if widths else None


@dataclass
class Instance:
    name: str
    depot: Location
    vehicles: list[Vehicle]
    requests: list[Request]
    horizon_days: int = 4
    current_day: Optional[int] = None  # "today"; defaults to the last arrival day
    shift_start: float = 480.0  # 08:00
    depot_close: float = 1200.0  # 20:00
    minutes_per_unit: float = 1.0
    max_lead_days: int = 2
    # optional explicit travel cost: {from_node: {to_node: distance}} over node ids
    # ("depot", "<request_id>:P", "<request_id>:D"). None => Euclidean distance.
    distance_matrix: Optional[dict[str, dict[str, float]]] = None
    synthetic: bool = False
    description: str = ""

    def __post_init__(self):
        if self.current_day is None and self.requests:
            self.current_day = max((r.arrival_day for r in self.requests), default=0)
        elif self.current_day is None:
            self.current_day = 0

    # ---- node helpers -------------------------------------------------
    @staticmethod
    def node_id(request_id: str, kind: str) -> str:
        return f"{request_id}:{kind}"

    def request_map(self) -> dict[str, Request]:
        return {r.id: r for r in self.requests}

    def vehicle_map(self) -> dict[str, Vehicle]:
        return {v.id: v for v in self.vehicles}

    def node_locations(self) -> dict[str, Location]:
        nodes = {DEPOT_NODE: self.depot}
        for r in self.requests:
            nodes[self.node_id(r.id, PICKUP)] = r.pickup
            nodes[self.node_id(r.id, DELIVERY)] = r.delivery
        return nodes

    def received_requests(self) -> list[Request]:
        """Requests received up to and including ``current_day`` in arrival order."""
        reqs = [r for r in self.requests if r.arrival_day <= self.current_day]
        return sorted(reqs, key=lambda r: (r.arrival_day, r.arrival_seq, r.id))

    def zips(self) -> list[str]:
        zs = {self.depot.zip} if self.depot.zip else set()
        for r in self.requests:
            zs.update(z for z in (r.pickup.zip, r.delivery.zip) if z)
        return sorted(zs)

    # ---- serialisation ------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        def loc(x):
            return Location(float(x["x"]), float(x["y"]), str(x.get("zip", "")))

        def tw(x):
            return None if x is None else TimeWindow(float(x["earliest"]), float(x["latest"]))

        vehicles = [Vehicle(str(v["id"]), int(v["capacity"]), float(v.get("shift_duration", 480.0))) for v in d["vehicles"]]
        requests = [
            Request(
                id=str(r["id"]),
                arrival_day=int(r["arrival_day"]),
                arrival_seq=int(r.get("arrival_seq", i)),
                pickup=loc(r["pickup"]),
                delivery=loc(r["delivery"]),
                demand=int(r.get("demand", 1)),
                priority=int(r.get("priority", NORMAL)),
                pickup_tw=tw(r.get("pickup_tw")),
                delivery_tw=tw(r.get("delivery_tw")),
                service_time=float(r.get("service_time", 0.0)),
                customer_id=r.get("customer_id"),
            )
            for i, r in enumerate(d["requests"])
        ]
        return Instance(
            name=str(d.get("name", "instance")),
            depot=loc(d["depot"]),
            vehicles=vehicles,
            requests=requests,
            horizon_days=int(d.get("horizon_days", 4)),
            current_day=d.get("current_day"),
            shift_start=float(d.get("shift_start", 480.0)),
            depot_close=float(d.get("depot_close", 1200.0)),
            minutes_per_unit=float(d.get("minutes_per_unit", 1.0)),
            max_lead_days=int(d.get("max_lead_days", 2)),
            distance_matrix=d.get("distance_matrix"),
            synthetic=bool(d.get("synthetic", False)),
            description=str(d.get("description", "")),
        )


# ---- plans -----------------------------------------------------------
@dataclass(frozen=True)
class Stop:
    request_id: str
    kind: str  # "P" or "D"

    @property
    def node(self) -> str:
        return Instance.node_id(self.request_id, self.kind)


@dataclass
class Route:
    day: int
    vehicle_id: str
    stops: list[Stop] = field(default_factory=list)


@dataclass
class Decision:
    request_id: str
    status: str  # "accepted" | "rejected"
    day: Optional[int] = None
    vehicle_id: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class PolicyParams:
    """Parameters of one run of the cheapest-insertion heuristic (one candidate plan).

    The paper does not say how its m alternative solutions differ (NOT SPECIFIED); here they are
    produced by varying these policy parameters (and an optional cost-noise seed).
    """

    label: str = "baseline"
    day_penalty: float = 0.0  # cost added per day of delay w.r.t. the arrival day
    max_detour: Optional[float] = None  # skip insertions whose raw added distance exceeds this
    tight_tw_threshold: Optional[float] = None  # reject requests with a window narrower than this (min)
    shift_limit: Optional[float] = None  # extra cap on route duration (min)
    reserve_fraction: float = 0.0  # share of capacity reserved for high-priority load
    avoid_zip: Optional[str] = None  # avoid service in this ZIP during the peak window
    peak_start: float = 690.0  # 11:30
    peak_end: float = 810.0  # 13:30
    balance_weight: float = 0.0  # cost added per minute of the route's current duration
    noise: float = 0.0  # multiplicative noise on adjusted cost (0 = deterministic cheapest)
    noise_seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    plan_id: str
    policy: PolicyParams
    routes: list[Route]
    decisions: dict[str, Decision]
    feasible: Optional[bool] = None  # set ONLY by constraints.finalize_plan
    violations: list[str] = field(default_factory=list)

    def accepted_ids(self) -> list[str]:
        return [rid for rid, d in self.decisions.items() if d.status == "accepted"]

    def rejected_ids(self) -> list[str]:
        return [rid for rid, d in self.decisions.items() if d.status == "rejected"]

    def fingerprint(self) -> str:
        parts = []
        for r in sorted(self.routes, key=lambda r: (r.day, r.vehicle_id)):
            parts.append(f"{r.day}|{r.vehicle_id}|" + ",".join(f"{s.request_id}{s.kind}" for s in r.stops))
        parts.append("R:" + ",".join(sorted(self.rejected_ids())))
        return ";".join(parts)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "policy": self.policy.to_dict(),
            "feasible": self.feasible,
            "violations": list(self.violations),
            "routes": [
                {"day": r.day, "vehicle_id": r.vehicle_id, "stops": [{"request_id": s.request_id, "kind": s.kind} for s in r.stops]}
                for r in self.routes
            ],
            "decisions": {k: asdict(v) for k, v in self.decisions.items()},
        }

    @staticmethod
    def from_dict(d: dict) -> "Plan":
        pol = PolicyParams(**d["policy"])
        routes = [Route(int(r["day"]), r["vehicle_id"], [Stop(s["request_id"], s["kind"]) for s in r["stops"]]) for r in d["routes"]]
        decisions = {k: Decision(**v) for k, v in d["decisions"].items()}
        return Plan(d["plan_id"], pol, routes, decisions, d.get("feasible"), list(d.get("violations", [])))


def is_finite_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
