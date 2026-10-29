"""End-to-end pipeline: routing -> candidate plans -> (DMG) -> evaluation matrix -> MCDM.

This is the single entry point used by the CLI, the experiments and the Streamlit UI (the UI contains no business
logic). Hard feasibility is decided by the deterministic validator only; metrics come from verified specifications or
the reference implementations; the LLM only generates metric specifications and chooses/justifies a plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .candidate_generation import CandidateSet, generate_candidates
from .config import RESULTS_DIR
from .dmg.evaluation import EvalError, evaluate_spec
from .dmg.examples import get_examples
from .dmg.generation import DMGPrompt, DMGResult, run_dmg
from .llm.base import LLMProvider
from .mcdm.selection import Candidate, MCDMResult, run_mcdm
from .plan_view import build_plan_view
from .problem_instance import DistanceModel, ensure_valid
from .reference_metrics import ALL_METRICS, PAPER_METRICS, compute_metric
from .runlog import append_log, new_run_id, run_dir, utc_now, write_json
from .schemas import Instance, Plan


@dataclass
class MetricRequest:
    """A natural-language request for a new metric (handled by DMG)."""

    text: str
    oracle: Optional[str] = None  # reference metric id (e.g. 'z5') whose values are the expected results; None = execution-only test
    metric_id: Optional[str] = None  # assigned automatically (d1, d2, ...)


@dataclass
class PipelineResult:
    run_id: str
    created_at: str
    instance: Instance
    provider_info: dict
    category: str
    candidate_set: CandidateSet
    matrix: dict[str, dict[str, float]]  # plan_id -> metric id -> value (feasible plans only)
    metric_ids: list[str]
    dmg_results: dict[str, DMGResult]
    dynamic_metric_texts: dict[str, str]
    mcdm: Optional[MCDMResult]
    preference: str
    timings: dict[str, float]
    out_dir: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def plans(self) -> list[Plan]:
        return self.candidate_set.plans

    def to_record(self) -> dict:
        return {
            "run_id": self.run_id, "created_at": self.created_at, "category": self.category, "provider": self.provider_info,
            "instance": {"name": self.instance.name, "synthetic": self.instance.synthetic, "n_requests": len(self.instance.requests),
                         "n_vehicles": len(self.instance.vehicles), "horizon_days": self.instance.horizon_days, "current_day": self.instance.current_day},
            "requested_m": self.candidate_set.requested_m, "produced_m": self.candidate_set.produced_m, "seed": self.candidate_set.seed,
            "n_feasible": sum(1 for p in self.plans if p.feasible), "metric_ids": self.metric_ids, "preference": self.preference,
            "dmg": {k: {"status": v.status, "attempts_used": v.attempts_used, "error": v.error} for k, v in self.dmg_results.items()},
            "mcdm": ({"status": self.mcdm.status, "selected_plan_id": self.mcdm.selected_plan_id, "error": self.mcdm.error} if self.mcdm else None),
            "timings_s": self.timings, "warnings": self.warnings,
        }


def classify_category(provider: LLMProvider, inst: Instance) -> str:
    """Result category (docs/experiment_protocol.md). A mock provider or synthetic data can never be 'original'."""
    if provider.is_mock or inst.synthetic:
        return "synthetic_demonstration"
    return "exploratory"


def run_pipeline(
    inst: Instance,
    preference: str,
    provider: LLMProvider,
    *,
    metric_requests: Optional[list[MetricRequest]] = None,
    m: int = 10,
    seed: int = 0,
    max_attempts: int = 5,
    include_extension_metrics: bool = True,
    save: bool = True,
    run_id: Optional[str] = None,
    results_dir: Optional[Path] = None,
    log_path: Optional[Path] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """Run the full workflow. Raises ``InstanceValidationError`` for invalid instances (before any computation)."""
    say = progress or (lambda msg: None)
    ensure_valid(inst)
    run_id = run_id or new_run_id()
    created = utc_now()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    append_log("run_started", run_id, log_path, provider=provider.info(), instance=inst.name, m=m, seed=seed)

    t0 = time.perf_counter()
    say(f"Generating {m} candidate plans with the cheapest-insertion heuristic ...")
    dm = DistanceModel(inst)
    cset = generate_candidates(inst, m, seed=seed, dm=dm)
    warnings.extend(cset.notes)
    timings["candidate_generation_s"] = time.perf_counter() - t0

    views = {p.plan_id: build_plan_view(inst, p, dm=dm) for p in cset.plans}
    base_ids = (ALL_METRICS if include_extension_metrics else PAPER_METRICS)
    matrix: dict[str, dict[str, float]] = {}
    for p in cset.plans:
        if p.feasible:
            matrix[p.plan_id] = {mid: compute_metric(mid, views[p.plan_id]) for mid in base_ids}
        else:
            warnings.append(f"plan {p.plan_id} failed the deterministic validator and is excluded: {p.violations[:2]}")
    metric_ids = list(base_ids)

    dmg_results: dict[str, DMGResult] = {}
    dyn_texts: dict[str, str] = {}
    t1 = time.perf_counter()
    examples = get_examples()
    for i, req in enumerate(metric_requests or [], start=1):
        mid = req.metric_id or f"d{i}"
        say(f"DMG: generating metric {mid} ...")
        expected = None
        if req.oracle:
            expected = [compute_metric(req.oracle, e.view) for e in examples]
        res = run_dmg(DMGPrompt(req.text, [e.view for e in examples], expected), provider, max_attempts)
        dmg_results[mid] = res
        dyn_texts[mid] = req.text
        if not res.success or res.spec is None:
            warnings.append(f"DMG failed for {mid}: {res.warning_to_supervisor}")
            continue
        if expected is None:
            warnings.append(f"metric {mid} was validated without expected results (execution-only test)")
        try:
            vals = {pid: evaluate_spec(res.spec, views[pid]) for pid in matrix}
        except EvalError as exc:
            warnings.append(f"metric {mid} could not be evaluated on candidate plans: {exc}")
            continue
        for pid, v in vals.items():
            matrix[pid][mid] = v
        metric_ids.append(mid)
    timings["dmg_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    mcdm_res: Optional[MCDMResult] = None
    if matrix:
        say("MCDM: selecting the preferred plan ...")
        cands = [Candidate(pid, vals, True) for pid, vals in matrix.items()]
        mcdm_res = run_mcdm(cands, preference, provider, metric_ids, extra_descriptions=dyn_texts)
        if not mcdm_res.success:
            warnings.append(f"MCDM failed: {mcdm_res.warning_to_supervisor}")
    else:
        warnings.append("no feasible candidate plan: MCDM was not run")
    timings["mcdm_s"] = time.perf_counter() - t2
    timings["total_s"] = time.perf_counter() - t0

    res = PipelineResult(run_id, created, inst, provider.info(), classify_category(provider, inst), cset, matrix, metric_ids, dmg_results,
                         dyn_texts, mcdm_res, preference, timings, warnings=warnings)
    if save:
        res.out_dir = save_run(res, results_dir)
    append_log("run_finished", run_id, log_path, status=("ok" if mcdm_res and mcdm_res.success else "incomplete"),
               selected=(mcdm_res.selected_plan_id if mcdm_res else None), timings=timings, warnings=len(warnings))
    return res


def save_run(res: PipelineResult, results_dir: Optional[Path] = None) -> Path:
    d = run_dir(res.run_id, "runs", results_dir or RESULTS_DIR)
    write_json(d / "record.json", res.to_record())
    write_json(d / "instance.json", res.instance.to_dict())
    write_json(d / "candidates.json", [p.to_dict() for p in res.plans])
    write_json(d / "matrix.json", {"metric_ids": res.metric_ids, "rows": res.matrix})
    write_json(d / "dmg_trace.json", {k: v.to_dict() for k, v in res.dmg_results.items()})
    write_json(d / "mcdm_trace.json", res.mcdm.to_dict() if res.mcdm else None)
    try:
        import pandas as pd

        pd.DataFrame.from_dict(res.matrix, orient="index").to_csv(d / "matrix.csv", index_label="plan_id")
    except Exception:  # noqa: BLE001 - CSV is a convenience copy
        pass
    return d
