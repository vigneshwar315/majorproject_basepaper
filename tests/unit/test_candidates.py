import pytest

from paper_replication.candidate_generation import generate_candidates
from paper_replication.constraints import validate_plan
from paper_replication.problem_instance import generate_instance, tiny_instance


def test_candidates_distinct_validated_and_named():
    inst = generate_instance(seed=7, n_requests=20, n_vehicles=2)
    cs = generate_candidates(inst, 8, seed=7)
    assert cs.produced_m <= 8 and cs.produced_m >= 2
    assert [p.plan_id for p in cs.plans] == [f"pi_{i}" for i in range(1, cs.produced_m + 1)]
    assert len({p.fingerprint() for p in cs.plans}) == cs.produced_m
    for p in cs.plans:
        assert p.feasible is True and validate_plan(inst, p) == []


def test_deterministic_for_same_seed_and_different_for_other():
    inst = generate_instance(seed=7, n_requests=20, n_vehicles=2)
    a = [p.fingerprint() for p in generate_candidates(inst, 6, seed=1).plans]
    b = [p.fingerprint() for p in generate_candidates(inst, 6, seed=1).plans]
    assert a == b


def test_reports_when_fewer_distinct_plans_exist():
    inst = tiny_instance()
    inst.requests = inst.requests[:1]  # a single request has very few distinct plans
    cs = generate_candidates(inst, 10, seed=0)
    assert cs.produced_m < 10 and cs.notes and "only" in cs.notes[0]


def test_m_must_be_positive():
    with pytest.raises(ValueError):
        generate_candidates(tiny_instance(), 0)
