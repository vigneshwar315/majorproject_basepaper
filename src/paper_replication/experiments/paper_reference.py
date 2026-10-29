"""Loader for the paper's reported values (kept separate from anything this project computes)."""

from __future__ import annotations

import json
from functools import lru_cache

from ..config import DATA_DIR


@lru_cache(maxsize=1)
def load_paper_reference() -> dict:
    return json.loads((DATA_DIR / "paper_reference_values.json").read_text(encoding="utf-8"))


def table9_matrix() -> tuple[list[str], dict[str, dict[str, float]]]:
    """The paper's own evaluation matrix (Table 9) as plan_id -> metric -> value. PAPER-PROVIDED INPUT, not our output."""
    t = load_paper_reference()["table9_example_evaluation_matrix"]
    ids = t["metric_ids"]
    return ids, {pid: dict(zip(ids, vals)) for pid, vals in t["rows"].items()}
