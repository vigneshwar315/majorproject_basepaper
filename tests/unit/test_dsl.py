import json

import pytest

from paper_replication.dmg.evaluation import EvalError, evaluate_spec
from paper_replication.dmg.examples import get_examples
from paper_replication.dmg.metric_specification import (
    MAX_DEPTH,
    MAX_NODES,
    SpecError,
    parse_spec,
    validate_spec_dict,
)
from paper_replication.dmg.reference_specs import EXPRESSIONS, reference_spec
from paper_replication.reference_metrics import ALL_METRICS, compute_metric


def _spec(expr, name="m"):
    return {"name": name, "description": "", "expr": expr}


def test_reference_specs_agree_with_reference_metrics_on_examples():
    for mid in ALL_METRICS:
        spec = reference_spec(mid)
        for ex in get_examples():
            assert evaluate_spec(spec, ex.view) == pytest.approx(compute_metric(mid, ex.view), abs=1e-9), (mid, ex.name)


def test_hand_verified_example_3_values():
    ex3 = get_examples()[2]
    assert compute_metric("z3", ex3.view) == pytest.approx(172.882, abs=1e-3)
    assert compute_metric("z5", ex3.view) == pytest.approx(190.0, abs=1e-6)
    assert compute_metric("z1", ex3.view) == pytest.approx(4 / 5)


def test_all_expected_values_are_finite_numbers():
    import math

    for mid in ALL_METRICS:
        for ex in get_examples():
            assert math.isfinite(compute_metric(mid, ex.view))


@pytest.mark.parametrize("bad", [
    {"op": "eval", "args": []},
    {"op": "__import__", "args": [{"op": "const", "value": 1}]},
    {"op": "field", "name": "accepted"},  # field outside an aggregation
    {"op": "agg", "fn": "sum", "over": "requests"},  # missing value
    {"op": "agg", "fn": "sum", "over": "nothing", "value": {"op": "const", "value": 1}},
    {"op": "agg", "fn": "median3", "over": "routes", "value": {"op": "field", "name": "distance"}},
    {"op": "agg", "fn": "sum", "over": "routes", "value": {"op": "field", "name": "no_such_field"}},
    {"op": "ctx", "name": "secret"},
    {"op": "add", "args": [{"op": "const", "value": 1}]},
    {"op": "const", "value": "x" * 100},
    {"op": "const", "value": float("inf")},
    {"op": "const", "value": [1]},
    {"op": "if", "cond": {"op": "const", "value": 1}, "then": {"op": "const", "value": 1}, "else": {"op": "const", "value": 2}},
])
def test_invalid_expressions_rejected(bad):
    assert validate_spec_dict(_spec(bad)), bad


def test_unexpected_keys_and_nesting_rejected():
    inner = {"op": "agg", "fn": "sum", "over": "routes", "value": {"op": "field", "name": "distance"}}
    nested = {"op": "agg", "fn": "sum", "over": "routes", "value": inner}
    assert any("nested" in e for e in validate_spec_dict(_spec(nested)))
    assert any("unexpected keys" in e for e in validate_spec_dict(_spec({"op": "const", "value": 1, "code": "import os"})))


def test_limits_on_depth_and_size():
    e = {"op": "const", "value": 1}
    for _ in range(MAX_DEPTH + 2):
        e = {"op": "neg", "args": [e]}
    assert any("deeper" in x for x in validate_spec_dict(_spec(e)))
    wide = {"op": "add", "args": [{"op": "const", "value": 1} for _ in range(MAX_NODES + 5)]}
    assert any("nodes" in x for x in validate_spec_dict(_spec(wide)))


def test_parse_errors_are_spec_errors_not_crashes():
    for src in ("not json", "[1,2]", "42", json.dumps({"name": "x"})):
        with pytest.raises(SpecError):
            parse_spec(src)


def test_code_fence_tolerated_and_python_never_executed(tmp_path):
    marker = tmp_path / "pwned"
    payload = f"__import__('pathlib').Path({str(marker)!r}).write_text('x')"
    with pytest.raises(SpecError):
        parse_spec(payload)
    with pytest.raises(SpecError):
        parse_spec(json.dumps(_spec({"op": "const", "value": payload, "x": payload})))
    assert not marker.exists()
    ok = "```json\n" + json.dumps(_spec(EXPRESSIONS["z3"])) + "\n```"
    assert parse_spec(ok).expr == EXPRESSIONS["z3"]


def test_empty_aggregations_are_zero_and_division_is_safe():
    for mid in ("z1", "z3", "z6", "z7"):
        assert evaluate_spec(reference_spec(mid), {**get_examples()[0].view, "requests": [], "routes": [], "visits": []}) == 0.0
    spec = parse_spec(_spec({"op": "safe_div", "args": [{"op": "const", "value": 1}, {"op": "const", "value": 0}]}))
    assert evaluate_spec(spec, get_examples()[0].view) == 0.0
    div = parse_spec(_spec({"op": "div", "args": [{"op": "const", "value": 1}, {"op": "const", "value": 0}]}))
    with pytest.raises(EvalError):
        evaluate_spec(div, get_examples()[0].view)


def test_fingerprint_ignores_name_and_description():
    a = parse_spec(_spec(EXPRESSIONS["z3"], "a"))
    b = parse_spec({"name": "b", "description": "other", "expr": EXPRESSIONS["z3"]})
    assert a.fingerprint() == b.fingerprint()
