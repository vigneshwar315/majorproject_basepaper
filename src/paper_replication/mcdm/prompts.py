"""MCDM prompt construction. Prompt wording is ours (the paper's prompts are NOT AVAILABLE)."""

from __future__ import annotations

from ..reference_metrics import DESCRIPTIONS, DIRECTION

MCDM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_plan_id": {"type": "string", "description": "id of exactly one candidate plan from the evaluation matrix"},
        "justification": {"type": "string", "description": "why this plan best satisfies the preferences, citing the metric values"},
        "criteria_used": {"type": "array", "items": {"type": "string"}, "description": "metric ids (e.g. z1) that drove the decision"},
    },
    "required": ["selected_plan_id", "justification", "criteria_used"],
}

SELECT_SYSTEM = (
    "You are the multi-criteria decision-making module of a vehicle-routing decision support system. You receive an "
    "evaluation matrix (one row per candidate plan, one column per metric) and the supervisor's natural-language trade-off "
    "preferences. Choose exactly ONE candidate plan that best satisfies the preferences. Rules: (1) selected_plan_id must be "
    "an id that appears in the matrix; (2) when plans tie on the criteria named in the preferences, prefer the plan that "
    "performs better on the remaining metrics; (3) in the justification cite only metric values that appear in the matrix, "
    "and explain which trade-offs you accepted. Reply with the requested JSON object only."
)


def _fmt(x: float) -> str:
    return f"{x:.4g}" if abs(x) < 1e5 else f"{x:.0f}"


def matrix_text(candidates: list[dict], metric_ids: list[str]) -> str:
    """Compact text rendering of the evaluation matrix (candidates: dicts with plan_id and metrics)."""
    lines = ["plan_id | " + " | ".join(metric_ids)]
    for c in candidates:
        lines.append(f"{c['plan_id']} | " + " | ".join(_fmt(c["metrics"][m]) for m in metric_ids))
    return "\n".join(lines)


def metric_legend(metric_ids: list[str], extra_descriptions: dict[str, str] | None = None) -> str:
    desc = {**DESCRIPTIONS, **(extra_descriptions or {})}
    lines = []
    for m in metric_ids:
        d = DIRECTION.get(m)
        hint = "higher is better" if d == "max" else "lower is better" if d == "min" else "direction: follow the supervisor preferences"
        lines.append(f"{m} ({hint}): {desc.get(m, '')}")
    return "\n".join(lines)


def build_select_user(preference: str, candidates: list[dict], metric_ids: list[str], previous_error: str = "", extra_descriptions: dict[str, str] | None = None) -> str:
    parts = [
        f"SUPERVISOR TRADE-OFF PREFERENCES:\n{preference}",
        f"METRICS:\n{metric_legend(metric_ids, extra_descriptions)}",
        f"EVALUATION MATRIX ({len(candidates)} candidate plans):\n{matrix_text(candidates, metric_ids)}",
    ]
    if previous_error:
        parts.append(f"YOUR PREVIOUS ANSWER WAS REJECTED: {previous_error}\nAnswer again, following the rules.")
    return "\n\n".join(parts)
