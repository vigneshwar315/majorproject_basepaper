"""Comprehension-verification building blocks for DMG (paper Sec. III-C).

(a) validation of the generated artefact against the description (an LLM check here, plus deterministic static
    checks that we add; hard constraints are never delegated to an LLM),
(b) execution on representative examples and comparison with expected results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..llm.base import LLMError, LLMProvider, LLMRequest
from .evaluation import EvalError, evaluate_spec
from .metric_specification import MetricSpec, SpecError, parse_spec

VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["match", "mismatch"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

VALIDATE_SYSTEM = (
    "You are a meticulous reviewer of metric specifications for a vehicle-routing decision support system. "
    "Given a natural-language metric description and a JSON metric specification, decide whether the specification "
    "computes exactly the described quantity. Answer verdict='match' only if it does; otherwise 'mismatch' and explain "
    "what differs. Do not rewrite the specification."
)


@dataclass
class StaticResult:
    ok: bool
    spec: Optional[MetricSpec] = None
    errors: list[str] = field(default_factory=list)


def static_validate(spec_json: str) -> StaticResult:
    try:
        return StaticResult(True, parse_spec(spec_json), [])
    except SpecError as exc:
        return StaticResult(False, None, exc.errors)


@dataclass
class CheckResult:
    ok: bool
    verdict: str
    reason: str
    meta: dict = field(default_factory=dict)
    error: Optional[str] = None


def llm_comprehension_check(provider: LLMProvider, metric_text: str, spec: MetricSpec) -> CheckResult:
    """Ask the provider whether ``spec`` corresponds to ``metric_text`` (paper III-C step a)."""
    user = f"METRIC DESCRIPTION:\n{metric_text}\n\nMETRIC SPECIFICATION (JSON):\n{spec.to_json(indent=2)}"
    req = LLMRequest(
        task="dmg_validate", system=VALIDATE_SYSTEM, user=user, json_schema=VALIDATION_SCHEMA, schema_name="validation",
        payload={"metric": metric_text, "spec_json": spec.to_json()},
    )
    resp = provider.generate_structured(req)
    verdict = str(resp.data.get("verdict", "")).lower()
    reason = str(resp.data.get("reason", ""))
    if verdict not in ("match", "mismatch"):
        return CheckResult(False, verdict or "invalid", "validation reply did not contain a valid verdict", resp.meta(), "invalid verdict")
    return CheckResult(verdict == "match", verdict, reason, resp.meta(), None if verdict == "match" else f"validation: {reason}")


@dataclass
class ExecutionResult:
    ok: bool
    values: list[Optional[float]] = field(default_factory=list)
    expected: list[float] = field(default_factory=list)
    error: Optional[str] = None


def values_close(a: float, b: float, rel: float = 1e-6, abs_: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def execute_on_examples(spec: MetricSpec, examples: list[dict], expected: Optional[list[float]]) -> ExecutionResult:
    """Run the spec on every example view and compare with the expected values.

    ``expected=None`` (no oracle supplied) only checks that the metric runs and yields finite numbers - a weaker test
    that callers must label as such.
    """
    if expected is None:
        vals = []
        for i, view in enumerate(examples):
            try:
                vals.append(evaluate_spec(spec, view))
            except (EvalError, SpecError) as exc:
                return ExecutionResult(False, vals + [None], [], f"run-time error on example {i + 1}: {exc}")
        return ExecutionResult(True, vals, [], None)
    if len(examples) != len(expected):
        return ExecutionResult(False, [], list(expected), "number of examples and expected results differ")
    vals: list[Optional[float]] = []
    for i, view in enumerate(examples):
        try:
            vals.append(evaluate_spec(spec, view))
        except (EvalError, SpecError) as exc:
            vals.append(None)
            return ExecutionResult(False, vals, list(expected), f"run-time error on example {i + 1}: {exc}")
    bad = [i + 1 for i, (v, e) in enumerate(zip(vals, expected)) if v is None or not values_close(v, e)]
    if bad:
        detail = "; ".join(f"example {i}: got {vals[i - 1]:.6g}, expected {expected[i - 1]:.6g}" for i in bad)
        return ExecutionResult(False, vals, list(expected), f"results do not match expected values ({detail})")
    return ExecutionResult(True, vals, list(expected), None)


__all__ = ["CheckResult", "ExecutionResult", "LLMError", "StaticResult", "execute_on_examples", "llm_comprehension_check", "static_validate"]
