"""Frozen fingerprints. They detect *unintended change*, they are NOT ground truth (see docs/execution_log.md).

Regenerate deliberately with:  PYTHONPATH=src python tests/regression/make_baseline.py
"""

import json
from pathlib import Path

import pytest

from paper_replication.candidate_generation import generate_candidates
from paper_replication.plan_view import build_plan_view
from paper_replication.problem_instance import generate_instance
from paper_replication.reference_metrics import PAPER_METRICS, compute_all

BASELINE = Path(__file__).parent / "baseline.json"


def _compute():
    inst = generate_instance(seed=20251028, n_requests=24, n_vehicles=3)
    cs = generate_candidates(inst, 6, seed=20251028)
    out = {"fingerprints": [p.fingerprint() for p in cs.plans], "metrics": {}}
    for p in cs.plans:
        out["metrics"][p.plan_id] = {k: round(v, 6) for k, v in compute_all(build_plan_view(inst, p), PAPER_METRICS).items()}
    return out


@pytest.mark.skipif(not BASELINE.exists(), reason="baseline.json not generated")
def test_plans_and_metrics_match_frozen_baseline():
    frozen = json.loads(BASELINE.read_text())
    now = _compute()
    assert now["fingerprints"] == frozen["fingerprints"]
    for pid, vals in frozen["metrics"].items():
        for k, v in vals.items():
            assert now["metrics"][pid][k] == pytest.approx(v, abs=1e-6), (pid, k)
