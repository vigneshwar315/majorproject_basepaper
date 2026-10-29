"""DMG with comprehension verification (paper Algorithm 1, p.149782).

Deviations (see docs/uncertainties_and_deviations.md): the LLM emits a constrained metric *specification* instead of
executable code (U01); the loop runs at most ``max_attempts`` passes (text reading, U11); deterministic static checks
are added before the LLM validation (U12). Prompt wording is ours (the paper's prompts are NOT AVAILABLE).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..llm.base import (
    LLMAuthError,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMRequestError,
    ModelNotAvailableError,
)
from ..redact import redact
from .metric_specification import SPEC_JSON_SCHEMA, MetricSpec, dsl_reference
from .validation import execute_on_examples, llm_comprehension_check, static_validate

FATAL_ERRORS = (LLMAuthError, LLMRequestError, ModelNotAvailableError)

GENERATE_SYSTEM = (
    "You translate a natural-language request for a new performance metric of a vehicle-routing plan into a JSON metric "
    "specification written in a small, restricted expression language. Reply with a JSON object {\"spec_json\": \"<the "
    "specification as a JSON string>\"}. Use ONLY the operators, entities and fields of the language reference. If you "
    "receive a previous specification and an error message, correct the specification so that the error disappears.\n\n"
    "LANGUAGE REFERENCE\n" + dsl_reference()
)


@dataclass
class DMGPrompt:
    """Paper's prompt object: fields ``metric``, ``examples`` and ``expected_results`` (p.149781)."""

    metric: str
    examples: list[dict]
    expected_results: Optional[list[float]]  # None = no oracle: execution-only test (weaker; must be labelled)


@dataclass
class AttemptTrace:
    attempt: int
    spec_json: str = ""
    generation_meta: dict = field(default_factory=dict)
    static_errors: list[str] = field(default_factory=list)
    llm_verdict: Optional[str] = None
    llm_reason: str = ""
    validation_meta: dict = field(default_factory=dict)
    execution_values: list = field(default_factory=list)
    execution_error: Optional[str] = None
    outcome: str = "pending"  # generation_error | static_invalid | validation_failed | execution_failed | success
    error: Optional[str] = None
    generation_s: float = 0.0
    validation_s: float = 0.0
    execution_s: float = 0.0


@dataclass
class DMGResult:
    status: str  # "Success" | "Failure"
    spec: Optional[MetricSpec]
    attempts_used: int
    max_attempts: int
    trace: list[AttemptTrace]
    error: Optional[str] = None
    warning_to_supervisor: Optional[str] = None
    provider: str = ""
    model: str = ""
    is_mock: bool = False

    @property
    def success(self) -> bool:
        return self.status == "Success"

    def totals(self) -> dict:
        gen = [t.generation_meta for t in self.trace if t.generation_meta]
        val = [t.validation_meta for t in self.trace if t.validation_meta]

        def tok(ms, key):
            vals = [m.get(key) for m in ms]
            return None if not vals or any(v is None for v in vals) else sum(vals)

        return {
            "generation_input_tokens": tok(gen, "input_tokens"), "generation_output_tokens": tok(gen, "output_tokens"),
            "validation_input_tokens": tok(val, "input_tokens"), "validation_output_tokens": tok(val, "output_tokens"),
            "generation_s": sum(t.generation_s for t in self.trace), "validation_s": sum(t.validation_s for t in self.trace),
        }

    def to_dict(self) -> dict:
        from dataclasses import asdict

        d = {
            "status": self.status, "attempts_used": self.attempts_used, "max_attempts": self.max_attempts, "error": self.error,
            "warning_to_supervisor": self.warning_to_supervisor, "provider": self.provider, "model": self.model,
            "is_mock": self.is_mock, "spec": self.spec.to_dict() if self.spec else None,
            "trace": [asdict(t) for t in self.trace], "totals": self.totals(),
        }
        return d


def build_generate_user_prompt(metric: str, previous_spec: str, error: str) -> str:
    parts = [f"METRIC REQUEST:\n{metric}"]
    if previous_spec:
        parts.append(f"PREVIOUS SPECIFICATION (it failed):\n{previous_spec}")
    if error:
        parts.append(f"ERROR FROM THE PREVIOUS ATTEMPT:\n{error}")
        parts.append("Fix the specification.")
    return "\n\n".join(parts)


def run_dmg(prompt: DMGPrompt, provider: LLMProvider, max_attempts: int = 5) -> DMGResult:
    """Algorithm 1: generate -> validate -> execute on examples -> compare, with bounded self-debugging."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    attempts = 0
    spec_json, error = "", ""
    trace: list[AttemptTrace] = []
    base = dict(provider=provider.name, model=provider.model, is_mock=provider.is_mock, max_attempts=max_attempts)
    while attempts < max_attempts:
        tr = AttemptTrace(attempt=attempts + 1)
        trace.append(tr)
        # --- generate(prompt.metric, code, error)
        t0 = time.perf_counter()
        try:
            req = LLMRequest(
                task="dmg_generate", system=GENERATE_SYSTEM, user=build_generate_user_prompt(prompt.metric, spec_json, error),
                json_schema=SPEC_JSON_SCHEMA, schema_name="metric_spec",
                payload={"metric": prompt.metric, "previous_spec": spec_json, "error": error, "attempt": attempts},
            )
            resp = provider.generate_structured(req)
            tr.generation_meta = resp.meta()
            tr.spec_json = str(resp.data.get("spec_json", ""))
        except FATAL_ERRORS as exc:
            tr.generation_s = time.perf_counter() - t0
            tr.outcome, tr.error = "generation_error", redact(exc)
            return DMGResult("Failure", None, attempts + 1, error=tr.error,
                             warning_to_supervisor=f"DMG aborted: {tr.error}", trace=trace, **base)
        except LLMError as exc:
            tr.generation_s = time.perf_counter() - t0
            tr.outcome, tr.error = "generation_error", redact(exc)
            error, attempts = tr.error, attempts + 1
            continue
        tr.generation_s = time.perf_counter() - t0
        # --- validate(prompt.metric, code): static checks, then LLM comprehension check
        static = static_validate(tr.spec_json)
        if not static.ok or static.spec is None:
            tr.static_errors = static.errors
            tr.outcome, tr.error = "static_invalid", "invalid specification: " + "; ".join(static.errors)
            spec_json, error, attempts = tr.spec_json, tr.error, attempts + 1
            continue
        t1 = time.perf_counter()
        try:
            check = llm_comprehension_check(provider, prompt.metric, static.spec)
        except FATAL_ERRORS as exc:
            tr.validation_s = time.perf_counter() - t1
            tr.outcome, tr.error = "generation_error", redact(exc)
            return DMGResult("Failure", None, attempts + 1, error=tr.error,
                             warning_to_supervisor=f"DMG aborted: {tr.error}", trace=trace, **base)
        except LLMError as exc:
            tr.validation_s = time.perf_counter() - t1
            tr.outcome, tr.error = "validation_failed", f"validation call failed: {redact(exc)}"
            spec_json, error, attempts = tr.spec_json, tr.error, attempts + 1
            continue
        tr.validation_s = time.perf_counter() - t1
        tr.llm_verdict, tr.llm_reason, tr.validation_meta = check.verdict, check.reason, check.meta
        if not check.ok:
            tr.outcome, tr.error = "validation_failed", check.error or "validation failed"
            spec_json, error, attempts = tr.spec_json, tr.error, attempts + 1
            continue
        # --- execute(code, prompt.examples) and compare with prompt.expected_results
        t2 = time.perf_counter()
        ex = execute_on_examples(static.spec, prompt.examples, prompt.expected_results)
        tr.execution_s = time.perf_counter() - t2
        tr.execution_values = ex.values
        if ex.ok:
            tr.outcome = "success"
            return DMGResult("Success", static.spec, attempts + 1, trace=trace, **base)
        tr.outcome, tr.execution_error, tr.error = "execution_failed", ex.error, ex.error
        spec_json, error, attempts = tr.spec_json, ex.error or "execution failed", attempts + 1
    return DMGResult(
        "Failure", None, attempts, error=error,
        warning_to_supervisor=("DMG could not produce a validated metric within max_attempts. The metric prompt may be imprecise or "
                               "incomplete; please clarify it (add a formula or pseudocode) and retry. Last error: " + error),
        trace=trace, **base,
    )
