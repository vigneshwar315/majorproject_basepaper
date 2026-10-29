import math

import pytest

from paper_replication.problem_instance import (
    DistanceModel,
    ensure_valid,
    generate_instance,
    tiny_instance,
    validate_instance,
)
from paper_replication.schemas import Instance, InstanceValidationError, Location, Request, Vehicle


def test_tiny_instance_is_valid():
    assert validate_instance(tiny_instance()) == []


def test_generator_is_deterministic_and_synthetic():
    a = generate_instance(seed=3, n_requests=12, n_vehicles=2)
    b = generate_instance(seed=3, n_requests=12, n_vehicles=2)
    assert a.to_dict() == b.to_dict()
    assert a.synthetic is True
    assert validate_instance(a) == []
    assert generate_instance(seed=4, n_requests=12, n_vehicles=2).to_dict() != a.to_dict()


def test_round_trip_serialisation():
    inst = tiny_instance()
    again = Instance.from_dict(inst.to_dict())
    assert again.to_dict() == inst.to_dict()


def test_empty_requests_is_valid():
    inst = tiny_instance()
    inst.requests = []
    assert validate_instance(inst) == []


def test_no_vehicles_is_invalid():
    inst = tiny_instance()
    inst.vehicles = []
    assert any("vehicle" in e for e in validate_instance(inst))


def test_duplicate_request_ids_rejected():
    inst = tiny_instance()
    inst.requests.append(inst.requests[0])
    errs = validate_instance(inst)
    assert any("duplicate request id" in e for e in errs)
    with pytest.raises(InstanceValidationError):
        ensure_valid(inst)


def test_invalid_numbers_reported():
    inst = tiny_instance()
    inst.vehicles = [Vehicle("V1", 0, -5.0)]
    inst.requests[0] = Request("a", 0, 0, Location(math.nan, 0), Location(1, 1), demand=0, priority=7)
    errs = " | ".join(validate_instance(inst))
    for needle in ("capacity", "shift_duration", "demand", "priority", "finite"):
        assert needle in errs


def test_inverted_time_window_rejected():
    from paper_replication.schemas import TimeWindow

    inst = tiny_instance()
    r = inst.requests[2]
    inst.requests[2] = Request(r.id, r.arrival_day, r.arrival_seq, r.pickup, r.delivery, pickup_tw=TimeWindow(700, 600))
    assert any("earliest must be <= latest" in e for e in validate_instance(inst))


def test_missing_travel_cost_detected():
    inst = tiny_instance()
    nodes = list(inst.node_locations())
    inst.distance_matrix = {a: {b: 1.0 for b in nodes if b != a} for a in nodes}
    assert validate_instance(inst) == []
    del inst.distance_matrix["a:P"]["a:D"]
    errs = validate_instance(inst)
    assert any("missing entry" in e and "a:P" in e for e in errs)


def test_explicit_matrix_is_used():
    inst = tiny_instance()
    nodes = list(inst.node_locations())
    inst.distance_matrix = {a: {b: 7.0 for b in nodes if b != a} for a in nodes}
    dm = DistanceModel(inst)
    assert dm.dist("depot", "a:P") == 7.0
    assert dm.dist("a:P", "a:P") == 0.0


def test_euclidean_distance_and_minutes_per_unit():
    inst = tiny_instance()
    inst.minutes_per_unit = 2.0
    dm = DistanceModel(inst)
    assert dm.dist("depot", "a:P") == pytest.approx(5.0)
    assert dm.travel("depot", "a:P") == pytest.approx(10.0)
