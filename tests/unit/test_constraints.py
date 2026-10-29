"""Hand-computed timing and constraint cases. Tiny instance: depot (0,0), V1/V2 capacity 3, shift 08:00-20:00."""

import pytest

from paper_replication.constraints import evaluate_route, finalize_plan, plan_route_results, validate_plan
from paper_replication.problem_instance import DistanceModel, tiny_instance
from paper_replication.schemas import (
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

P, D = "P", "D"


def _eval(inst, vehicle_id, stops, policy=None):
    return evaluate_route(inst, DistanceModel(inst), inst.request_map(), inst.vehicle_map()[vehicle_id], stops, policy=policy)


def test_single_request_timing_by_hand():
    inst = tiny_instance()
    r = _eval(inst, "V1", [Stop("a", P), Stop("a", D)])
    # depot->(3,4)=5, (3,4)->(6,8)=5, (6,8)->depot=10
    assert r.feasible, r.violations
    assert r.distance == pytest.approx(20.0)
    assert r.depart == pytest.approx(480.0)
    assert r.end == pytest.approx(500.0)
    assert r.duration == pytest.approx(20.0)
    assert r.waiting == 0.0
    assert r.stops[0].arrival == pytest.approx(485.0)


def test_capacity_exceeded():
    inst = tiny_instance()
    # a(1)+b(2)=3 fits (capacity 3); adding c(1) before any delivery exceeds
    ok = _eval(inst, "V1", [Stop("a", P), Stop("b", P), Stop("a", D), Stop("b", D)])
    assert ok.feasible and ok.max_load == 3
    bad = _eval(inst, "V1", [Stop("a", P), Stop("b", P), Stop("c", P), Stop("c", D), Stop("a", D), Stop("b", D)])
    assert not bad.feasible and any("capacity exceeded" in v for v in bad.violations)


def test_precedence_delivery_before_pickup():
    inst = tiny_instance()
    r = _eval(inst, "V1", [Stop("a", D), Stop("a", P)])
    assert not r.feasible and any("delivery before pickup" in v for v in r.violations)


def test_pickup_without_delivery():
    inst = tiny_instance()
    r = _eval(inst, "V1", [Stop("a", P)])
    assert not r.feasible and any("never delivered" in v for v in r.violations)


def test_time_window_violation_and_waiting():
    depot = Location(0, 0)
    req_late = Request("l", 0, 0, Location(10, 0), Location(10, 1), pickup_tw=TimeWindow(480, 485))
    inst = Instance("tw", depot, [Vehicle("V1", 2, 480.0)], [req_late], horizon_days=2, current_day=0, shift_start=480.0, depot_close=1200.0)
    # depart = max(480, 480-10)=480; arrival 490 > latest 485
    r = _eval(inst, "V1", [Stop("l", P), Stop("l", D)])
    assert not r.feasible and any("time window violated" in v for v in r.violations)
    # window opens later than arrival: vehicle leaves late (no artificial wait at the start)
    req_early = Request("e", 0, 0, Location(10, 0), Location(10, 1), pickup_tw=TimeWindow(600, 700))
    inst2 = Instance("tw2", depot, [Vehicle("V1", 2, 480.0)], [req_early], horizon_days=2, current_day=0)
    r2 = _eval(inst2, "V1", [Stop("e", P), Stop("e", D)])
    assert r2.feasible and r2.depart == pytest.approx(590.0) and r2.stops[0].start == pytest.approx(600.0) and r2.waiting == 0.0


def test_waiting_between_stops_is_counted():
    depot = Location(0, 0)
    reqs = [Request("x", 0, 0, Location(10, 0), Location(20, 0)), Request("y", 0, 1, Location(30, 0), Location(30, 5), pickup_tw=TimeWindow(700, 800))]
    inst = Instance("wait", depot, [Vehicle("V1", 3, 600.0)], reqs, horizon_days=2, current_day=0)
    r = _eval(inst, "V1", [Stop("x", P), Stop("x", D), Stop("y", P), Stop("y", D)])
    # arrive at y pickup at 480+10+10+10=510 -> wait 190 until 700
    assert r.feasible, r.violations
    assert r.waiting == pytest.approx(190.0)
    assert r.stops[2].start == pytest.approx(700.0)


def test_shift_and_depot_close_violations():
    depot = Location(0, 0)
    req = Request("f", 0, 0, Location(200, 0), Location(300, 0))
    inst = Instance("far", depot, [Vehicle("V1", 2, 480.0)], [req], horizon_days=2, current_day=0)
    r = _eval(inst, "V1", [Stop("f", P), Stop("f", D)])  # duration 600 > shift 480
    assert not r.feasible and any("exceeds shift" in v for v in r.violations)
    inst2 = Instance("far2", depot, [Vehicle("V1", 2, 5000.0)], [req], horizon_days=2, current_day=0, depot_close=900.0)
    r2 = _eval(inst2, "V1", [Stop("f", P), Stop("f", D)])
    assert not r2.feasible and any("after depot close" in v for v in r2.violations)


def test_policy_violations_are_labelled():
    inst = tiny_instance()
    r = _eval(inst, "V1", [Stop("a", P), Stop("a", D)], policy=PolicyParams(shift_limit=10.0))
    assert not r.feasible and any(v.startswith("POLICY:") for v in r.violations)


def _plan(inst, routes, accepted, rejected=()):
    dec = {rid: Decision(rid, "accepted", day=d, vehicle_id=v) for rid, (d, v) in accepted.items()}
    dec.update({rid: Decision(rid, "rejected", reason="test") for rid in rejected})
    return Plan("t", PolicyParams(), routes, dec)


def test_valid_multi_vehicle_multi_day_plan():
    inst = tiny_instance()
    plan = _plan(
        inst,
        [Route(0, "V1", [Stop("a", P), Stop("a", D)]), Route(0, "V2", [Stop("b", P), Stop("b", D)]), Route(1, "V1", [Stop("c", P), Stop("c", D)])],
        {"a": (0, "V1"), "b": (0, "V2"), "c": (1, "V1")},
    )
    assert finalize_plan(inst, plan).feasible is True
    assert set(plan_route_results(inst, plan)) == {(0, "V1"), (0, "V2"), (1, "V1")}


def test_validator_catches_plan_level_errors():
    inst = tiny_instance()
    # request a scheduled outside its lead window (arrival 0, lead 1 -> day 2 not allowed), c missing from decisions
    plan = _plan(inst, [Route(2, "V1", [Stop("a", P), Stop("a", D)])], {"a": (2, "V1")}, rejected=["b"])
    v = validate_plan(inst, plan)
    assert any("outside [0, 1]" in x for x in v)
    assert any("decisions must cover exactly the received requests" in x for x in v)
    assert finalize_plan(inst, plan).feasible is False


def test_validator_catches_split_pickup_delivery_and_duplicates():
    inst = tiny_instance()
    plan = _plan(inst, [Route(0, "V1", [Stop("a", P)]), Route(0, "V2", [Stop("a", D)])], {"a": (0, "V1")}, rejected=["b", "c"])
    v = " | ".join(validate_plan(inst, plan))
    assert "different routes" in v
    dup = _plan(inst, [Route(0, "V1", [Stop("a", P), Stop("a", D)]), Route(0, "V1", [])], {"a": (0, "V1")}, rejected=["b", "c"])
    assert any("duplicate route" in x for x in validate_plan(inst, dup))


def test_unknown_vehicle_and_request():
    inst = tiny_instance()
    plan = _plan(inst, [Route(0, "VX", [Stop("a", P)]), Route(0, "V1", [Stop("zz", P)])], {}, rejected=["a", "b", "c"])
    v = " | ".join(validate_plan(inst, plan))
    assert "unknown vehicle" in v and "unknown request" in v


def test_feasible_flag_only_set_by_finalize():
    inst = tiny_instance()
    plan = _plan(inst, [], {}, rejected=["a", "b", "c"])
    assert plan.feasible is None
    assert finalize_plan(inst, plan).feasible is True
