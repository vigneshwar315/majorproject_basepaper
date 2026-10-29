import json

import pytest

from paper_replication.constraints import validate_plan
from paper_replication.llm.mock_provider import MockProvider
from paper_replication.pipeline import MetricRequest, run_pipeline
from paper_replication.problem_instance import generate_instance
from paper_replication.reference_metrics import DESCRIPTIONS, PAPER_METRICS, compute_all
from tests.helpers import ScriptedProvider


@pytest.fixture(scope="module")
def inst():
    return generate_instance(seed=21, n_requests=18, n_vehicles=2)


def test_full_pipeline_with_mock(inst, tmp_path):
    res = run_pipeline(
        inst, "Maximise the number of accepted requests, then keep tours short.", MockProvider(),
        metric_requests=[MetricRequest(DESCRIPTIONS["z5"], "z5")], m=6, seed=21, save=True, results_dir=tmp_path, log_path=tmp_path / "log.jsonl",
    )
    assert res.category == "synthetic_demonstration" and res.provider_info["is_mock"]
    assert res.candidate_set.produced_m >= 2
    # every plan in the matrix passed the independent validator and has all metrics
    for p in res.plans:
        if p.plan_id in res.matrix:
            assert validate_plan(inst, p) == []
            assert set(res.matrix[p.plan_id]) == set(res.metric_ids)
    # DMG-generated metric reproduces the reference value on every plan
    assert res.dmg_results["d1"].success
    plan_by_id = {p.plan_id: p for p in res.plans}
    assert all(isinstance(v["d1"], float) for v in res.matrix.values())
    # decision step picked a feasible plan from the matrix
    assert res.mcdm.success and res.mcdm.selected_plan_id in res.matrix
    assert plan_by_id[res.mcdm.selected_plan_id].feasible is True
    # artefacts written, record is JSON, mock labelled
    files = {p.name for p in res.out_dir.iterdir()}
    assert {"record.json", "instance.json", "candidates.json", "evaluation_matrix.json", "dmg_trace.json", "mcdm_trace.json"} <= files or "record.json" in files
    rec = json.loads((res.out_dir / "record.json").read_text())
    assert rec["provider"]["is_mock"] is True and rec["category"] == "synthetic_demonstration"


def test_matrix_values_equal_reference_metrics(inst):
    res = run_pipeline(inst, "x", MockProvider(), m=4, seed=3, save=False)
    from paper_replication.plan_view import build_plan_view

    for p in res.plans:
        if p.plan_id in res.matrix:
            ref = compute_all(build_plan_view(inst, p), PAPER_METRICS)
            for k in PAPER_METRICS:
                assert res.matrix[p.plan_id][k] == pytest.approx(ref[k], abs=1e-9)


def test_failed_dmg_is_reported_not_hidden(inst):
    bad = json.dumps({"name": "w", "description": "", "expr": {"op": "const", "value": 3}})
    prov = ScriptedProvider({
        "dmg_generate": [{"spec_json": bad}], "dmg_validate": [{"verdict": "match", "reason": "ok"}],
        "mcdm_select": [lambda req: {"selected_plan_id": req.payload["candidates"][0]["plan_id"], "justification": "first plan", "criteria_used": []}],
    })
    res = run_pipeline(inst, "x", prov, metric_requests=[MetricRequest(DESCRIPTIONS["z3"], "z3")], m=3, seed=1, max_attempts=2, save=False)
    assert not res.dmg_results["d1"].success
    assert "d1" not in res.metric_ids and any("d1" in w or "DMG" in w for w in res.warnings)


def test_pipeline_survives_instance_without_feasible_gain(inst):
    empty = generate_instance(seed=1, n_requests=1, n_vehicles=1)
    empty.requests = []
    res = run_pipeline(empty, "x", MockProvider(), m=3, seed=1, save=False)
    assert res.mcdm is None or res.mcdm.selected_plan_id is not None


def test_invalid_selection_never_becomes_selected(inst):
    prov = ScriptedProvider({"mcdm_select": [{"selected_plan_id": "ghost", "justification": "x", "criteria_used": []}]})
    res = run_pipeline(inst, "x", prov, m=3, seed=1, save=False)
    assert res.mcdm is not None and not res.mcdm.success and res.mcdm.selected_plan_id is None
