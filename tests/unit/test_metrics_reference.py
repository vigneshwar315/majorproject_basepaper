import pytest

from paper_replication.dmg.examples import get_examples
from paper_replication.reference_metrics import (
    ALL_METRICS,
    DIRECTION,
    EXTENSION_METRICS,
    PAPER_METRICS,
    compute_all,
    compute_metric,
)


def test_metric_inventory():
    assert PAPER_METRICS == [f"z{i}" for i in range(1, 11)]
    assert EXTENSION_METRICS == [f"z{i}" for i in range(11, 15)]
    assert set(DIRECTION) == set(ALL_METRICS) and set(DIRECTION.values()) <= {"max", "min"}


def test_example_3_hand_values():
    v = get_examples()[2].view
    # requests x,y,z,v,w -> 4 accepted of 5; today=day 1 gets z,v,w -> 2 of 3 accepted
    assert compute_metric("z1", v) == pytest.approx(0.8)
    assert compute_metric("z2", v) == pytest.approx(2 / 3)
    assert compute_metric("z3", v) == pytest.approx(172.882, abs=1e-3)
    assert compute_metric("z5", v) == pytest.approx(190.0)
    assert compute_metric("z10", v) == pytest.approx(5 + 5 + 10 + 5)


def test_examples_distinguish_metrics():
    ex = get_examples()
    for m in ALL_METRICS:
        values = {round(compute_metric(m, e.view), 6) for e in ex}
        assert len(values) >= 2, f"{m} does not separate the three examples"


def test_compute_all_subset():
    v = get_examples()[0].view
    assert set(compute_all(v, ["z1", "z3"])) == {"z1", "z3"}
