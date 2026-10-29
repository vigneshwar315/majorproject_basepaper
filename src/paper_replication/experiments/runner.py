"""Experiment runner for the paper's tests C1-C6 (Sec. IV-B).

Every experiment: (1) never runs automatically at start-up, (2) writes real artefacts to
``results/experiments/<ID>/<run_id>/`` (results.csv, summary.json, meta.json, status.json), (3) labels the run with a
result category, (4) keeps the paper's reported numbers out of the produced files (see ``reporting.py`` for comparisons).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ..candidate_generation import generate_candidates
from ..config import PROJECT_ROOT, RESULTS_DIR
from ..dmg.examples import expected_for, get_examples
from ..dmg.generation import DMGPrompt, run_dmg
from ..llm.base import LLMProvider
from ..mcdm.reference_rules import acceptable_plans, regret
from ..mcdm.selection import Candidate, run_mcdm
from ..plan_view import build_plan_view
from ..problem_instance import DistanceModel, generate_instance
from ..redact import redact
from ..reference_metrics import ALL_METRICS, compute_all
from ..runlog import append_log, new_run_id, utc_now, write_json
from .metrics import mean_or_none, rate, sum_or_none
from .prompt_sets import (
    TRADEOFF_SEEDS,
    degraded_metric_prompt_set,
    degraded_tradeoff_prompt_set,
    metric_prompt_set,
    tradeoff_prompt_set,
)

DEFAULT_SEED = 20251028
Progress = Callable[[int, int, str], None]


def experiment_category(provider: LLMProvider) -> str:
    """Never 'original_conditions': the paper's prompts, instances and LangChain set-up are not available."""
    if provider.is_mock:
        return "synthetic_demonstration"
    if provider.name == "openai" and "gpt-4o" in provider.model.lower():
        return "partial_reproduction"
    return "alternative_model"


@dataclass
class ExperimentSpec:
    exp_id: str
    title: str
    source: str
    question: str
    design: str
    caveats: list[str]
    defaults: dict[str, Any]
    fn: Callable[..., "tuple[pd.DataFrame, dict]"]


@dataclass
class ExperimentResult:
    exp_id: str
    run_id: str
    category: str
    rows: pd.DataFrame
    summary: dict
    meta: dict
    out_dir: Path


# ---------------------------------------------------------------------------------------------------------------
# shared building blocks
# ---------------------------------------------------------------------------------------------------------------
def _matrices(seed: int, n_matrices: int, m_values: list[int], inst_kwargs: Optional[dict] = None) -> dict[tuple[int, int], dict[str, dict[str, float]]]:
    """Evaluation matrices from REAL heuristic plans: for each matrix index k an instance (seed+k) and nested candidate
    sets (first m plans of a common set). Returns {(k, m): plan_id -> metric -> value} (feasible plans only)."""
    out: dict[tuple[int, int], dict[str, dict[str, float]]] = {}
    m_max = max(m_values)
    for k in range(n_matrices):
        inst = generate_instance(seed=seed + k, **(inst_kwargs or {}))
        dm = DistanceModel(inst)
        cs = generate_candidates(inst, m_max, seed=seed + k, dm=dm)
        full = {p.plan_id: compute_all(build_plan_view(inst, p, dm=dm)) for p in cs.plans if p.feasible}
        ids = list(full)
        for m in m_values:
            out[(k, m)] = {pid: full[pid] for pid in ids[:m]}
    return out


def _dmg_trial(provider: LLMProvider, text: str, metric_id: str, max_attempts: int) -> dict:
    ex = get_examples()
    res = run_dmg(DMGPrompt(text, [e.view for e in ex], expected_for(metric_id, ex)), provider, max_attempts)
    reached_exec = any(t.llm_verdict == "match" for t in res.trace)
    tot = res.totals()
    return {
        "status": res.status, "success": res.success, "reached_execution": reached_exec, "attempts_used": res.attempts_used,
        "outcomes": ">".join(t.outcome for t in res.trace), "error": redact(res.error or ""),
        "gen_in_tokens": tot["generation_input_tokens"], "gen_out_tokens": tot["generation_output_tokens"],
        "val_in_tokens": tot["validation_input_tokens"], "val_out_tokens": tot["validation_output_tokens"],
        "generation_s": tot["generation_s"], "validation_s": tot["validation_s"],
    }


def _mcdm_trial(provider: LLMProvider, matrix: dict[str, dict[str, float]], text: str, rule_no: int, metric_ids: list[str]) -> dict:
    cands = [Candidate(pid, vals, True) for pid, vals in matrix.items()]
    res = run_mcdm(cands, text, provider, metric_ids)
    acc = acceptable_plans(rule_no, matrix)
    sel = res.selected_plan_id
    rg = regret(rule_no, matrix, sel) if sel in matrix else {}
    v = res.validation
    return {
        "status": res.status, "selected": sel, "correct": bool(sel in acc) if sel else False, "n_acceptable": len(acc),
        "attempts": res.attempts, "valid_output": res.success, "error": redact(res.error or ""),
        "regret_z1": rg.get("z1"), "prompt_chars": res.prompt_chars, "latency_s": res.latency_s,
        "input_tokens": res.meta.get("input_tokens") if res.meta else None, "output_tokens": res.meta.get("output_tokens") if res.meta else None,
        "claims_checked": v.claims_checked if v else None, "claims_supported": v.claims_supported if v else None,
        "unsupported_claims": "; ".join(v.unsupported_claims) if v else "", "justification": res.justification,
    }


def _tick(progress: Optional[Progress], i: int, n: int, msg: str) -> None:
    if progress:
        progress(i, n, msg)


# ---------------------------------------------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------------------------------------------
def exp_c1(provider, seed, max_attempts, progress=None, **_):
    rows = []
    prompts = [p for t in ("T", "TF", "TFP") for p in metric_prompt_set(t)]
    for i, p in enumerate(prompts, start=1):
        r = _dmg_trial(provider, p.text, p.metric_id, max_attempts)
        rows.append({"prompt_id": p.prompt_id, "ptype": p.ptype, "metric_id": p.metric_id, "source": p.source, **r})
        _tick(progress, i, len(prompts), f"C1 {p.prompt_id}")
    df = pd.DataFrame(rows)
    summ = {"max_attempts": max_attempts, "per_type": {}}
    for t, g in df.groupby("ptype"):
        ok = g[g["success"]]
        summ["per_type"][t] = {
            "n_prompts": int(len(g)), "generated": int(g["reached_execution"].sum()), "tests_passed": int(g["success"].sum()),
            "avg_attempts": mean_or_none(ok["attempts_used"].tolist()),
            "failed_metric_ids": sorted(set(g[~g["success"]]["metric_id"])),
        }
    return df, summ


def exp_c3(provider, seed, max_attempts, progress=None, **_):
    rows = []
    jobs = [(t, lv) for t in ("T", "TF", "TFP") for lv in ("informal", "unclear", "ambiguous")]
    total = 30 * len(jobs)
    n = 0
    for t, lv in jobs:
        for p in degraded_metric_prompt_set(t, lv):
            r = _dmg_trial(provider, p.text, p.metric_id, max_attempts)
            rows.append({"prompt_id": p.prompt_id, "ptype": t, "degradation": lv, "metric_id": p.metric_id, "source": p.source, **r})
            n += 1
            _tick(progress, n, total, f"C3 {p.prompt_id}")
    df = pd.DataFrame(rows)
    summ = {"max_attempts": max_attempts, "cells": {}}
    for (t, lv), g in df.groupby(["ptype", "degradation"]):
        summ["cells"].setdefault(t, {})[lv] = {"n_prompts": int(len(g)), "generated": int(g["reached_execution"].sum()), "passed": int(g["success"].sum())}
    return df, summ


def exp_c2(provider, seed, max_attempts, progress=None, n_matrices=3, ms=(10, 50, 100), **_):
    ms = [int(x) for x in ms]
    mats = _matrices(seed, n_matrices, ms)
    prompts = tradeoff_prompt_set()
    rows = []
    total = len(prompts) * len(mats)
    n = 0
    for (k, m), matrix in mats.items():
        ids = ALL_METRICS
        for p in prompts:
            r = _mcdm_trial(provider, matrix, p.text, p.rule_no, ids)
            rows.append({"matrix_index": k, "m_requested": m, "m_feasible": len(matrix), "prompt_id": p.prompt_id, "rule_no": p.rule_no, "source": p.source, **r})
            n += 1
            _tick(progress, n, total, f"C2 m={m} {p.prompt_id}")
    df = pd.DataFrame(rows)
    summ = {"n_prompts": len(prompts), "n_matrices": n_matrices, "per_m": {}}
    for m, g in df.groupby("m_requested"):
        correct = int(g["correct"].sum())
        summ["per_m"][str(m)] = {
            "trials": int(len(g)), "correct": correct, "correct_per_30_mean": correct / n_matrices,
            "invalid_or_failed": int((~g["valid_output"]).sum()), "mean_regret_z1_of_wrong": mean_or_none(g[~g["correct"]]["regret_z1"].tolist()),
        }
    return df, summ


def exp_c4(provider, seed, max_attempts, progress=None, n_matrices=3, m=10, **_):
    mats = _matrices(seed, n_matrices, [m])
    rows = []
    total = 3 * 30 * n_matrices
    n = 0
    for level in (1, 2, 3):
        for p in degraded_tradeoff_prompt_set(level):
            for k in range(n_matrices):
                r = _mcdm_trial(provider, mats[(k, m)], p.text, p.rule_no, ALL_METRICS)
                rows.append({"degradation_level": level, "matrix_index": k, "prompt_id": p.prompt_id, "rule_no": p.rule_no, "source": p.source, "text": p.text, **r})
                n += 1
                _tick(progress, n, total, f"C4 L{level} {p.prompt_id}")
    df = pd.DataFrame(rows)
    summ = {"m": m, "n_matrices": n_matrices, "assumption": "paper does not state m for Test C4; m=10 assumed", "per_level": {}}
    for lv, g in df.groupby("degradation_level"):
        c = int(g["correct"].sum())
        summ["per_level"][str(lv)] = {"trials": int(len(g)), "correct": c, "correct_per_30_mean": c / n_matrices, "invalid_or_failed": int((~g["valid_output"]).sum())}
    return df, summ


def _synthetic_matrix(base: dict[str, dict[str, float]], m: int, seed: int) -> dict[str, dict[str, float]]:
    """Bootstrap + 2% jitter of a real matrix to reach m rows (C5 only; clearly labelled SYNTHETIC)."""
    rng = np.random.default_rng(seed)
    keys = list(base)
    out = {}
    for i in range(m):
        src = base[keys[int(rng.integers(0, len(keys)))]]
        out[f"pi_{i + 1}"] = {k: float(v * rng.uniform(0.98, 1.02)) if k not in ("z1", "z2") else float(min(1.0, v * rng.uniform(0.99, 1.01))) for k, v in src.items()}
    return out


def exp_c5(provider, seed, max_attempts, progress=None, ms=(10, 50, 100, 200, 500, 1000), reps=3, **_):
    ms = [int(x) for x in ms]
    base = _matrices(seed, 1, [100])[(0, 100)]
    rows = []
    # DMG: three sample metrics, first T paraphrase, `reps` repetitions
    samples = [p for p in metric_prompt_set("T") if p.prompt_id.endswith("-1") and p.metric_id in ("z1", "z3", "z5")]
    total = len(samples) * reps + len(ms) * reps
    n = 0
    for p in samples:
        for r_i in range(reps):
            r = _dmg_trial(provider, p.text, p.metric_id, max_attempts)
            rows.append({"task": "DMG", "m": None, "matrix_kind": None, "rep": r_i, "prompt_id": p.prompt_id, **{k: r[k] for k in ("success", "attempts_used", "gen_in_tokens", "gen_out_tokens", "val_in_tokens", "val_out_tokens", "generation_s", "validation_s")}})
            n += 1
            _tick(progress, n, total, f"C5 DMG {p.prompt_id}")
    pref = TRADEOFF_SEEDS[6]
    for m in ms:
        for r_i in range(reps):
            matrix = dict(list(base.items())[:m]) if m <= len(base) else _synthetic_matrix(base, m, seed + m + r_i)
            kind = "real_heuristic_plans" if m <= len(base) else "SYNTHETIC_bootstrap"
            res = run_mcdm([Candidate(pid, v, True) for pid, v in matrix.items()], pref, provider, ALL_METRICS)
            rows.append({"task": "MCDM", "m": m, "matrix_kind": kind, "rep": r_i, "prompt_id": "TO6-seed", "success": res.success, "attempts_used": res.attempts,
                         "gen_in_tokens": res.meta.get("input_tokens") if res.meta else None, "gen_out_tokens": res.meta.get("output_tokens") if res.meta else None,
                         "val_in_tokens": None, "val_out_tokens": None, "generation_s": res.latency_s, "validation_s": None, "prompt_chars": res.prompt_chars})
            n += 1
            _tick(progress, n, total, f"C5 MCDM m={m}")
    df = pd.DataFrame(rows)
    summ: dict[str, Any] = {"reps": reps, "note": "token counts are provider-reported; None = unavailable. Mock provider reports no tokens and its wall-clock time is a mechanics check only.", "dmg": {}, "mcdm": {}}
    d = df[df["task"] == "DMG"]
    summ["dmg"] = {
        "runs": int(len(d)), "mean_generation_s": mean_or_none(d["generation_s"].tolist()), "mean_validation_s": mean_or_none(d["validation_s"].tolist()),
        "mean_generation_tokens": mean_or_none([sum_or_none([a, b]) for a, b in zip(d["gen_in_tokens"], d["gen_out_tokens"])]),
        "mean_validation_tokens": mean_or_none([sum_or_none([a, b]) for a, b in zip(d["val_in_tokens"], d["val_out_tokens"])]),
    }
    for m, g in df[df["task"] == "MCDM"].groupby("m"):
        summ["mcdm"][str(int(m))] = {
            "runs": int(len(g)), "matrix_kind": g["matrix_kind"].iloc[0], "mean_time_s": mean_or_none(g["generation_s"].tolist()),
            "mean_tokens": mean_or_none([sum_or_none([a, b]) for a, b in zip(g["gen_in_tokens"], g["gen_out_tokens"])]),
            "mean_prompt_chars_proxy": mean_or_none(g["prompt_chars"].tolist()), "success_rate": rate(int(g["success"].sum()), len(g)),
        }
    return df, summ


def exp_c6(provider, seed, max_attempts, progress=None, m=10, **_):
    matrix = _matrices(seed, 1, [m])[(0, m)]
    prompts = tradeoff_prompt_set()
    rows = []
    for i, p in enumerate(prompts, start=1):
        r = _mcdm_trial(provider, matrix, p.text, p.rule_no, ALL_METRICS)
        rows.append({"prompt_id": p.prompt_id, "rule_no": p.rule_no, "text": p.text, **{k: r[k] for k in ("status", "selected", "claims_checked", "claims_supported", "unsupported_claims", "justification")}})
        _tick(progress, i, len(prompts), f"C6 {p.prompt_id}")
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "Success"]
    with_claims = ok[ok["claims_checked"].fillna(0) > 0]
    summ = {
        "m": m, "decisions": int(len(df)), "valid_decisions": int(len(ok)), "with_numeric_claims": int(len(with_claims)),
        "fully_grounded": int((with_claims["claims_supported"] == with_claims["claims_checked"]).sum()),
        "with_unsupported_claims": int((with_claims["claims_supported"] < with_claims["claims_checked"]).sum()),
        "manual_review_required": True,
        "note": "The paper's C6 is a qualitative judgement by the authors. Here only numeric claims are checked automatically against the matrix; the full texts in results.csv require manual rating.",
    }
    return df, summ


_COMMON_CAVEATS = [
    "Instances are SYNTHETIC (the paper provides none) and the heuristic is a reconstruction (docs/uncertainties_and_deviations.md U03-U05).",
    "DMG produces a constrained metric specification, not executable code (U01).",
    "Prompt sets are hand-written paraphrases / rule-based degradations, not the paper's LLM-generated sets (U14).",
    "'Correct' MCDM selections mean agreement with our deterministic reference rules (U15).",
]

EXPERIMENTS: dict[str, ExperimentSpec] = {
    "C1": ExperimentSpec("C1", "Correctness of dynamic metrics generation", "Sec. IV-B1, Table 3", "Can DMG produce validated metrics from clearly articulated prompts (T/TF/TFP)?",
                         "10 seed metrics (Table 2) x 3 hand-written paraphrases = 30 prompts per type; each run through Algorithm 1 with 3 hand-verifiable examples.", _COMMON_CAVEATS, {"max_attempts": 5}, exp_c1),
    "C2": ExperimentSpec("C2", "Correctness of multi-criteria decision making", "Sec. IV-B2, Table 5", "Does MCDM select pi* according to the trade-off prompt as m grows (10/50/100)?",
                         "30 trade-off prompts (9 Table-4 seeds + paraphrases) x evaluation matrices from real heuristic plans (3 instances), m = 10/50/100.", _COMMON_CAVEATS, {"n_matrices": 3, "ms": (10, 50, 100)}, exp_c2),
    "C3": ExperimentSpec("C3", "Robustness to ill-defined metrics", "Sec. IV-B3, Table 6", "How does DMG degrade with informal, unclear and ambiguous metric prompts?",
                         "C1 prompts with their text part degraded by rules (informal/unclear/ambiguous); formulae/pseudocode kept.", _COMMON_CAVEATS, {"max_attempts": 5}, exp_c3),
    "C4": ExperimentSpec("C4", "Robustness to trade-off prompt clarity", "Sec. IV-B4, Table 7", "Does MCDM tolerate progressively less clear trade-off prompts?",
                         "C2 prompts degraded to levels 1-3 by rules; m = 10 assumed (not stated in the paper); 3 matrices.", _COMMON_CAVEATS, {"n_matrices": 3, "m": 10}, exp_c4),
    "C5": ExperimentSpec("C5", "Computational efficiency", "Sec. IV-B5, Table 8", "How do tokens and time scale for DMG and for MCDM with m = 10 ... 1000?",
                         "DMG on 3 sample metrics; MCDM with m = 10/50/100 (real plans) and 200/500/1000 (SYNTHETIC bootstrap matrices); 3 repetitions.", _COMMON_CAVEATS, {"ms": (10, 50, 100, 200, 500, 1000), "reps": 3}, exp_c5),
    "C6": ExperimentSpec("C6", "Interpretability of decisions", "Sec. IV-B6", "Are MCDM justifications consistent with the computed metrics?",
                         "C2 matrix (m = 10) and prompts; automatic check of numeric claims in the justification against the matrix; texts stored for manual rating.", _COMMON_CAVEATS, {"m": 10}, exp_c6),
}


def experiment_dir(exp_id: str, run_id: str, results_dir: Optional[Path] = None) -> Path:
    return (results_dir or RESULTS_DIR) / "experiments" / exp_id / run_id


def run_experiment(exp_id: str, provider: LLMProvider, *, seed: int = DEFAULT_SEED, max_attempts: int = 5, results_dir: Optional[Path] = None,
                   progress: Optional[Progress] = None, run_id: Optional[str] = None, log_path: Optional[Path] = None, **params) -> ExperimentResult:
    """Run one experiment explicitly and persist its artefacts. Raises on unrecoverable errors (status.json records them)."""
    if exp_id not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {exp_id!r}; available: {sorted(EXPERIMENTS)}")
    spec = EXPERIMENTS[exp_id]
    run_id = run_id or new_run_id(exp_id.lower())
    out = experiment_dir(exp_id, run_id, results_dir)
    category = experiment_category(provider)
    cfg = {**spec.defaults, **params, "seed": seed, "max_attempts": max_attempts}
    started = utc_now()
    t0 = time.perf_counter()
    write_json(out / "status.json", {"status": "running", "started_at": started, "exp_id": exp_id, "run_id": run_id})
    append_log("experiment_started", run_id, log_path, exp_id=exp_id, provider=provider.info(), config=cfg)
    try:
        df, summary = spec.fn(provider, seed, max_attempts, progress, **{k: v for k, v in cfg.items() if k not in ("seed", "max_attempts")})
    except Exception as exc:  # noqa: BLE001 - recorded and re-raised
        write_json(out / "status.json", {"status": "failed", "started_at": started, "finished_at": utc_now(), "error": redact(exc), "traceback": redact(traceback.format_exc(limit=5)), "exp_id": exp_id, "run_id": run_id})
        append_log("experiment_failed", run_id, log_path, exp_id=exp_id, error=redact(exc))
        raise
    elapsed = time.perf_counter() - t0
    meta = {
        "exp_id": exp_id, "title": spec.title, "source": spec.source, "run_id": run_id, "category": category, "provider": provider.info(),
        "config": cfg, "started_at": started, "finished_at": utc_now(), "elapsed_s": round(elapsed, 3), "caveats": spec.caveats,
        "paper_reference_file": "data/paper_reference_values.json", "mock_notice": ("MOCK PROVIDER - rule-based stand-in, NOT an LLM; results demonstrate pipeline mechanics only." if provider.is_mock else None),
        "project_root": str(PROJECT_ROOT.name),
    }
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    write_json(out / "summary.json", summary)
    write_json(out / "meta.json", meta)
    write_json(out / "status.json", {"status": "completed", "started_at": started, "finished_at": meta["finished_at"], "rows": int(len(df)), "exp_id": exp_id, "run_id": run_id})
    append_log("experiment_finished", run_id, log_path, exp_id=exp_id, rows=int(len(df)), elapsed_s=round(elapsed, 3), category=category)
    return ExperimentResult(exp_id, run_id, category, df, summary, meta, out)


def run_all(provider: LLMProvider, *, ids: Optional[list[str]] = None, **kw) -> list[ExperimentResult]:
    return [run_experiment(i, provider, **kw) for i in (ids or list(EXPERIMENTS))]
