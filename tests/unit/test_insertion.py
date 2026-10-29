import numpy as np
import pytest

from paper_replication.constraints import validate_plan
from paper_replication.insertion_heuristic import build_plan, insertion_cost_matrix
from paper_replication.problem_instance import DistanceModel, generate_instance, tiny_instance
from paper_replication.schemas import Instance, Location, PolicyParams, Request, Stop, TimeWindow, Vehicle


def test_cost_matrix_empty_route_by_hand():
    inst = tiny_instance()
    dm = DistanceModel(inst)
    m = insertion_cost_matrix(dm, [], "a:P", "a:D")
    assert m.shape == (1, 1)
    assert m[0, 0] == pytest.approx(20.0)  # 5 + 5 + 10 - 0


def test_cost_matrix_one_stop_by_hand():
    inst = tiny_instance()
    dm = DistanceModel(inst)
    stops = [Stop("b", "P"), Stop("b", "D")]
    m = insertion_cost_matrix(dm, stops, "a:P", "a:D")
    assert m.shape == (3, 3)
    assert np.isinf(m[1, 0])  # delivery position before pickup position is impossible
    # inserting a:P and a:D adjacent at the very end: (0,10)->(3,4)=sqrt(9+36), ->(6,8)=5, ->depot=10, minus (0,10)->depot=10
    expected = np.hypot(3, 6) + 5 + 10 - 10
    assert m[2, 2] == pytest.approx(expected)


def test_tiny_instance_plan_accepts_everything_and_is_feasible():
    inst = tiny_instance()
    plan = build_plan(inst, PolicyParams(), "p")
    assert validate_plan(inst, plan) == []
    assert plan.feasible is True
    assert set(plan.accepted_ids()) == {"a", "b", "c"}


def test_deterministic():
    inst = generate_instance(seed=11, n_requests=20, n_vehicles=2)
    assert build_plan(inst, PolicyParams(), "x").fingerprint() == build_plan(inst, PolicyParams(), "y").fingerprint()


def test_oversize_request_rejected_with_reason():
    inst = tiny_instance()
    inst.requests.append(Request("big", 1, 1, Location(1, 1), Location(2, 2), demand=9))
    plan = build_plan(inst, PolicyParams(), "p")
    assert "big" in plan.rejected_ids()
    assert plan.decisions["big"].reason
    assert plan.feasible is True


def test_tight_window_policy_rejects_tight_requests():
    inst = tiny_instance()
    inst.requests.append(Request("tight", 1, 1, Location(5, 5), Location(6, 6), pickup_tw=TimeWindow(700, 710)))
    base = build_plan(inst, PolicyParams(), "p")
    pol = build_plan(inst, PolicyParams(tight_tw_threshold=30.0), "q")
    assert "tight" in base.accepted_ids()
    assert "tight" in pol.rejected_ids()


def test_multi_vehicle_use_when_shift_forces_it():
    depot = Location(0, 0)
    # East and west requests: each alone is a ~45 minute round trip, both together ~90 minutes > the 60 minute shift
    reqs = [Request("east", 0, 0, Location(20, 0), Location(20, 5)), Request("west", 0, 1, Location(-20, 0), Location(-20, 5))]
    inst = Instance("cap", depot, [Vehicle("V1", 2, 60.0), Vehicle("V2", 2, 60.0)], reqs, horizon_days=1, current_day=0, max_lead_days=0)
    plan = build_plan(inst, PolicyParams(), "p")
    assert plan.feasible and set(plan.accepted_ids()) == {"east", "west"}
    assert {r.vehicle_id for r in plan.routes} == {"V1", "V2"}


def test_empty_instance_gives_empty_plan():
    inst = tiny_instance()
    inst.requests = []
    plan = build_plan(inst, PolicyParams(), "p")
    assert plan.routes == [] and plan.feasible is True
