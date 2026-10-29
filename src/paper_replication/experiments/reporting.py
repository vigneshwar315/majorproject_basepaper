"""Reading experiment artefacts, listing runs and comparing with the paper's reported values (only where valid)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import RESULTS_DIR
from .paper_reference import load_paper_reference

COMPARABLE_CATEGORIES = ("partial_reproduction", "alternative_model")


def _read_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_experiment_runs(results_dir: Optional[Path] = None) -> pd.DataFrame:
    """One row per run directory found on disk (real status.json / meta.json)."""
    base = (results_dir or RESULTS_DIR) / "experiments"
    rows = []
    if base.exists():
        for exp_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for run in sorted((p for p in exp_dir.iterdir() if p.is_dir()), reverse=True):
                st = _read_json(run / "status.json") or {}
                meta = _read_json(run / "meta.json") or {}
                rows.append({
                    "exp_id": exp_dir.name, "run_id": run.name, "status": st.get("status", "unknown"), "category": meta.get("category", ""),
                    "provider": (meta.get("provider") or {}).get("provider", ""), "model": (meta.get("provider") or {}).get("model", ""),
                    "is_mock": (meta.get("provider") or {}).get("is_mock"), "started_at": st.get("started_at", ""), "finished_at": st.get("finished_at", ""),
                })
    cols = ["exp_id", "run_id", "status", "category", "provider", "model", "is_mock", "started_at", "finished_at"]
    return pd.DataFrame(rows, columns=cols)


def load_experiment_run(exp_id: str, run_id: str, results_dir: Optional[Path] = None) -> dict:
    d = (results_dir or RESULTS_DIR) / "experiments" / exp_id / run_id
    csv = d / "results.csv"
    return {
        "dir": d, "status": _read_json(d / "status.json"), "meta": _read_json(d / "meta.json"), "summary": _read_json(d / "summary.json"),
        "results": pd.read_csv(csv) if csv.exists() else None,
    }


def status_counts(results_dir: Optional[Path] = None) -> dict[str, int]:
    df = list_experiment_runs(results_dir)
    counts = {"completed": 0, "failed": 0, "running": 0, "unknown": 0}
    for s in df["status"]:
        counts[s if s in counts else "unknown"] += 1
    counts["total"] = int(len(df))
    return counts


def comparison_with_paper(exp_id: str, summary: dict, category: str) -> tuple[pd.DataFrame, bool, str]:
    """Return (table, valid, reason). ``valid`` is False for mock/synthetic runs: values are then NOT placed side by side."""
    ref = load_paper_reference()
    rows: list[dict] = []
    if exp_id == "C1":
        for t, r in ref["table3_C1_dmg_accuracy"]["rows"].items():
            ours = summary.get("per_type", {}).get(t, {})
            for k, ok in (("generated", "generated"), ("avg_attempts", "avg_attempts"), ("tests_passed", "tests_passed")):
                rows.append({"item": f"Table 3 / {t}-prompts / {k}", "paper (of 30)": r[k], "this run": ours.get(ok)})
    elif exp_id == "C2":
        for m, c in ref["table5_C2_mcdm_accuracy"]["correct_by_m"].items():
            ours = summary.get("per_m", {}).get(m, {})
            rows.append({"item": f"Table 5 / m={m} / correct of 30", "paper (of 30)": c, "this run": ours.get("correct_per_30_mean")})
    elif exp_id == "C3":
        for t, lv in ref["table6_C3_robustness"]["rows"].items():
            for name, vals in lv.items():
                ours = summary.get("cells", {}).get(t, {}).get(name, {})
                rows.append({"item": f"Table 6 / {t} / {name} / passed (generated)", "paper (of 30)": f"{vals['passed']} ({vals['generated']})", "this run": f"{ours.get('passed')} ({ours.get('generated')})"})
    elif exp_id == "C4":
        for lv, c in ref["table7_C4_prompt_clarity"]["correct_by_degradation_level"].items():
            ours = summary.get("per_level", {}).get(lv, {})
            rows.append({"item": f"Table 7 / level {lv} / correct of 30", "paper (of 30)": c, "this run": ours.get("correct_per_30_mean")})
    elif exp_id == "C5":
        t8 = ref["table8_C5_tokens_time"]
        rows.append({"item": "Table 8 / DMG generation time (s)", "paper (of 30)": t8["dmg"]["code_generation"]["time_s"], "this run": summary.get("dmg", {}).get("mean_generation_s")})
        rows.append({"item": "Table 8 / DMG validation time (s)", "paper (of 30)": t8["dmg"]["validation"]["time_s"], "this run": summary.get("dmg", {}).get("mean_validation_s")})
        for m, v in t8["mcdm"].items():
            ours = summary.get("mcdm", {}).get(m, {})
            rows.append({"item": f"Table 8 / MCDM m={m} time (s)", "paper (of 30)": v["time_s"], "this run": ours.get("mean_time_s")})
            rows.append({"item": f"Table 8 / MCDM m={m} tokens", "paper (of 30)": v["tokens_per_run"], "this run": ours.get("mean_tokens")})
    else:
        return pd.DataFrame(), False, "The paper's C6 is a qualitative judgement; there is no numeric value to compare."
    df = pd.DataFrame(rows).rename(columns={"paper (of 30)": "paper reported"})
    if category not in COMPARABLE_CATEGORIES:
        return df, False, ("This run used the mock provider and/or synthetic data (category: %s). Its numbers measure pipeline mechanics, not language-model behaviour, "
                           "so they are NOT compared with the paper's GPT-4o results." % category)
    return df, True, ("Indicative comparison only: the paper's prompts, instances, heuristic details and model snapshot are unavailable; "
                      "metric definitions and ground-truth rules are our reconstructions (see docs/uncertainties_and_deviations.md).")
