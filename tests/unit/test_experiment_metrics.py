import math

from paper_replication.experiments.metrics import mean_or_none, rate, sum_or_none


def test_unavailable_values_stay_unavailable():
    nan = float("nan")
    assert mean_or_none([None, nan]) is None
    assert mean_or_none([1.0, None, nan, 3.0]) == 2.0
    assert sum_or_none([1, None]) is None and sum_or_none([nan, 1]) is None and sum_or_none([]) is None
    assert sum_or_none([1, 2]) == 3.0
    assert rate(1, 0) is None and rate(1, 4) == 0.25
    assert not math.isnan(mean_or_none([2.0]))
