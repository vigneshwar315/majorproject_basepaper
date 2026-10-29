"""Deterministic reference decision rules ("ground truth") for the seed trade-off prompts of Table 4.

The paper does not say how a selection was judged correct (NOT SPECIFIED, deviation U15). For each seed prompt we
define a lexicographic rule with tolerances; a selection is *correct* when it lies in the set of plans that survive
every stage. Accuracy reported by our experiments is therefore agreement with THIS rule, not with the paper's judgement.
Rules reference z11-z14 (extension metrics, not in Table 2) for prompts 3 and 8-10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..reference_metrics import DIRECTION


@dataclass(frozen=True)
class Stage:
    metric: str
    tol: float = 0.0  # keep plans within tol of the best value among survivors
    cap_factor: Optional[float] = None  # 'cap' stage: keep plans with metric <= min * cap_factor (min-metrics only)


RULES: dict[int, tuple[Stage, ...]] = {
    1: (Stage("z1", 0.01), Stage("z2", 0.005)),
    2: (Stage("z1", 0.01), Stage("z3", cap_factor=1.02), Stage("z2", 0.005)),
    3: (Stage("z1", 0.02), Stage("z11", 0.5)),
    5: (Stage("z1", 0.01), Stage("z2", 0.005), Stage("z8", 0.5)),
    6: (Stage("z1", 0.01), Stage("z2", 0.005), Stage("z6", 5.0)),
    7: (Stage("z1", 0.01), Stage("z10", 5.0)),
    8: (Stage("z1", 0.01), Stage("z12", 0.5)),
    9: (Stage("z1", 0.01), Stage("z13", 0.5)),
    10: (Stage("z1", 0.01), Stage("z14", 2.0)),
}


def acceptable_plans(rule_no: int, matrix: dict[str, dict[str, float]]) -> list[str]:
    """Plan ids that survive all stages of rule ``rule_no`` (matrix: plan_id -> metric -> value)."""
    surv = list(matrix)
    for st in RULES[rule_no]:
        if not surv:
            break
        vals = {p: matrix[p][st.metric] for p in surv}
        if st.cap_factor is not None:
            best = min(vals.values())
            surv = [p for p in surv if vals[p] <= best * st.cap_factor + 1e-12]
            continue
        if DIRECTION[st.metric] == "max":
            best = max(vals.values())
            surv = [p for p in surv if vals[p] >= best - st.tol - 1e-12]
        else:
            best = min(vals.values())
            surv = [p for p in surv if vals[p] <= best + st.tol + 1e-12]
    return surv


def regret(rule_no: int, matrix: dict[str, dict[str, float]], selected: str) -> dict[str, float]:
    """Per stage metric: signed gap between the selected plan and the best plan (>= 0 means worse than best).

    Measures how close to the optimum a not-acceptable selection is (cf. paper p.149784: 'remains close to the optimum').
    """
    out = {}
    for st in RULES[rule_no]:
        if st.cap_factor is not None:
            continue
        vals = [row[st.metric] for row in matrix.values()]
        sel = matrix[selected][st.metric]
        out[st.metric] = (max(vals) - sel) if DIRECTION[st.metric] == "max" else (sel - min(vals))
    return out


def is_correct(rule_no: int, matrix: dict[str, dict[str, float]], selected: Optional[str]) -> bool:
    return selected is not None and selected in acceptable_plans(rule_no, matrix)
