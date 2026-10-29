"""Three hand-verifiable example plans per metric (paper: each DMG prompt carries three examples + expected results).

The plans are written by hand (not produced by the heuristic) so that expected values can be checked on paper;
see tests/unit/test_metrics_reference.py for the hand calculations. ``expected_results`` come from the independent
reference implementations in ``reference_metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..constraints import finalize_plan
from ..plan_view import build_plan_view
from ..problem_instance import ensure_valid, tiny_instance
from ..reference_metrics import ALL_METRICS, compute_metric
from ..schemas import (
    DELIVERY,
    HIGH,
    NORMAL,
    PICKUP,
    Decision,
    Instance,
    Location,
    Plan,
    PolicyParams,
    Request,
    Route,
    Stop,
    TimeWindow,
    Vehicle,
)


@dataclass(frozen=True)
class DMGExample:
    name: str
    view: dict
    note: str


def _plan(inst: Instance, plan_id: str, routes: list[Route], accepted: dict[str, tuple[int, str]], rejected: list[str]) -> Plan:
    dec = {rid: Decision(rid, "accepted", day=d, vehicle_id=v) for rid, (d, v) in accepted.items()}
    dec.update({rid: Decision(rid, "rejected", reason="example") for rid in rejected})
    return finalize_plan(inst, Plan(plan_id, PolicyParams(label="hand-built"), routes, dec))


def _S(rid, kind):
    return Stop(rid, kind)


def _example_1() -> DMGExample:
    inst = tiny_instance()
    # widen c's window to 120 min so that this example has NO tight-window request (z11 must differ across examples)
    inst.requests = [(replace(r, pickup_tw=TimeWindow(600, 720)) if r.id == "c" else r) for r in inst.requests]
    plan = _plan(
        inst, "ex1",
        [Route(0, "V1", [_S("b", PICKUP), _S("a", PICKUP), _S("a", DELIVERY), _S("b", DELIVERY)]), Route(1, "V1", [_S("c", PICKUP), _S("c", DELIVERY)])],
        {"a": (0, "V1"), "b": (0, "V1"), "c": (1, "V1")}, [],
    )
    return DMGExample("example_1", build_plan_view(inst, plan, target_zip="00100"), "all three requests accepted; request c has a wide (120 min) pickup window")


def _example_2() -> DMGExample:
    """Long routes on consecutive days, a late return, a tight window and one customer served by two vehicles."""
    depot = Location(0, 0, "Z0")
    reqs = [
        Request("p", 0, 0, Location(150, 0, "Z1"), Location(150, 150, "Z1"), demand=1, priority=NORMAL, customer_id="cA"),
        Request("q", 0, 1, Location(20, 0, "Z2"), Location(20, 40, "Z2"), demand=1, priority=HIGH, customer_id="cA", pickup_tw=TimeWindow(1050, 1100)),
        Request("r", 1, 0, Location(150, 0, "Z1"), Location(150, 150, "Z1"), demand=1, priority=NORMAL, customer_id="cB"),
        Request("s", 1, 1, Location(10, 10, "Z3"), Location(20, 20, "Z3"), demand=9, priority=NORMAL, customer_id="cC"),  # too big => rejected
    ]
    inst = ensure_valid(Instance("example2", depot, [Vehicle("V1", 5, 600.0), Vehicle("V2", 5, 600.0)], reqs, horizon_days=4, current_day=1,
                                 max_lead_days=2, depot_close=1300.0, description="hand-built example: long routes, late return, tight window"))
    plan = _plan(
        inst, "ex2",
        [Route(1, "V1", [_S("p", PICKUP), _S("p", DELIVERY)]), Route(1, "V2", [_S("q", PICKUP), _S("q", DELIVERY)]), Route(2, "V1", [_S("r", PICKUP), _S("r", DELIVERY)])],
        {"p": (1, "V1"), "q": (1, "V2"), "r": (2, "V1")}, ["s"],
    )
    return DMGExample("example_2", build_plan_view(inst, plan, target_zip="Z1"), "long routes on consecutive days, late return (tight window), one customer with two drivers")


def _example_3() -> DMGExample:
    depot = Location(0, 0, "Z0")
    reqs = [
        Request("x", 0, 0, Location(10, 0, "Z1"), Location(20, 0, "Z1"), demand=1, priority=NORMAL),
        Request("y", 0, 1, Location(20, 10, "Z2"), Location(0, 10, "Z2"), demand=1, priority=NORMAL, pickup_tw=TimeWindow(700, 760)),
        Request("z", 1, 0, Location(5, 5, "Z1"), Location(5, 15, "Z1"), demand=1, priority=HIGH),
        Request("v", 1, 1, Location(30, 0, "Z3"), Location(40, 0, "Z3"), demand=1, priority=NORMAL),
        Request("w", 1, 2, Location(50, 50, "Z3"), Location(60, 50, "Z3"), demand=9, priority=NORMAL),  # exceeds capacity => rejected
    ]
    inst = ensure_valid(Instance("example3", depot, [Vehicle("V1", 4, 480.0), Vehicle("V2", 4, 480.0)], reqs, horizon_days=3, current_day=1,
                                 max_lead_days=1, description="hand-built example with waiting time, peak-hour visits and a rejection"))
    plan = _plan(
        inst, "ex3",
        [Route(0, "V1", [_S("x", PICKUP), _S("x", DELIVERY), _S("y", PICKUP), _S("y", DELIVERY)]),
         Route(1, "V1", [_S("z", PICKUP), _S("z", DELIVERY)]), Route(1, "V2", [_S("v", PICKUP), _S("v", DELIVERY)])],
        {"x": (0, "V1"), "y": (0, "V1"), "z": (1, "V1"), "v": (1, "V2")}, ["w"],
    )
    return DMGExample("example_3", build_plan_view(inst, plan, target_zip="Z2"), "waiting at y's pickup window (peak hours, ZIP Z2); request w rejected")


def get_examples() -> list[DMGExample]:
    return [_example_1(), _example_2(), _example_3()]


def expected_for(metric_id: str, examples: list[DMGExample] | None = None) -> list[float]:
    examples = examples or get_examples()
    return [compute_metric(metric_id, e.view) for e in examples]


def examples_payload(metric_id: str) -> tuple[list[dict], list[float]]:
    """(examples as JSON-able plan views, expected values) for one metric."""
    ex = get_examples()
    return [e.view for e in ex], expected_for(metric_id, ex)


__all__ = ["ALL_METRICS", "DMGExample", "examples_payload", "expected_for", "get_examples"]
