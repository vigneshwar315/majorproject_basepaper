import json

import pytest

from paper_replication.experiments.reporting import list_experiment_runs, load_experiment_run, status_counts
from paper_replication.experiments.runner import EXPERIMENTS, experiment_category, run_experiment
from paper_replication.llm.base import LLMProvider
from paper_replication.llm.mock_provider import MockProvider


def _run(exp, tmp_path, **kw):
    return run_experiment(exp, MockProvider(), results_dir=tmp_path, log_path=tmp_path / "log.jsonl", **kw)


def test_registry_covers_c1_to_c6():
    assert list(EXPERIMENTS) == ["C1", "C2", "C3", "C4", "C5", "C6"]


def test_category_never_original_and_mock_is_synthetic():
    assert experiment_category(MockProvider()) == "synthetic_demonstration"

    class Fake(LLMProvider):
        def __init__(self, name, model):
            super().__init__(model)
            self.name = name

        def _call(self, req): ...

    assert experiment_category(Fake("gemini", "g")) == "alternative_model"
    assert experiment_category(Fake("openai", "gpt-4o-2024")) == "partial_reproduction"
    assert experiment_category(Fake("openai", "gpt-5")) == "alternative_model"


def test_c1_artefacts_and_status(tmp_path):
    r = _run("C1", tmp_path)
    assert r.category == "synthetic_demonstration" and len(r.rows) == 90
    assert {p.name for p in r.out_dir.iterdir()} >= {"results.csv", "summary.json", "meta.json", "status.json"}
    assert json.loads((r.out_dir / "status.json").read_text())["status"] == "completed"
    assert set(r.summary["per_type"]) == {"T", "TF", "TFP"}
    runs = list_experiment_runs(tmp_path)
    assert len(runs) == 1 and runs.iloc[0]["status"] == "completed" and bool(runs.iloc[0]["is_mock"]) is True
    data = load_experiment_run("C1", r.run_id, tmp_path)
    assert data["results"] is not None and status_counts(tmp_path)["completed"] == 1


def test_c2_small_configuration(tmp_path):
    r = _run("C2", tmp_path, n_matrices=1, ms=(5, 10))
    assert set(r.summary["per_m"]) == {"5", "10"}
    assert len(r.rows) == 2 * 30
    assert r.rows["m_feasible"].max() <= 10


def test_c4_c5_c6_small(tmp_path):
    c4 = _run("C4", tmp_path, n_matrices=1, m=5)
    assert set(c4.summary["per_level"]) == {"1", "2", "3"}
    c5 = _run("C5", tmp_path, ms=(5, 10), reps=1)
    assert set(c5.summary["mcdm"]) == {"5", "10"}
    c6 = _run("C6", tmp_path, m=5)
    assert len(c6.rows) == 30


def test_failed_experiment_writes_failed_status(tmp_path):
    class Boom(MockProvider):
        def _call(self, req):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_experiment("C1", Boom(), results_dir=tmp_path, log_path=tmp_path / "l.jsonl", run_id="x1")
    st = json.loads((tmp_path / "experiments" / "C1" / "x1" / "status.json").read_text())
    assert st["status"] == "failed" and "boom" in st["error"]


def test_unknown_experiment(tmp_path):
    with pytest.raises(KeyError):
        _run("C9", tmp_path)


def test_c5_token_columns_are_unavailable_not_nan_for_mock(tmp_path):
    c5 = _run("C5", tmp_path, ms=(5,), reps=1)
    assert c5.summary["dmg"]["mean_generation_tokens"] is None
    assert c5.summary["mcdm"]["5"]["mean_tokens"] is None
    json.dumps(c5.summary, allow_nan=False)  # strict JSON: no NaN anywhere
