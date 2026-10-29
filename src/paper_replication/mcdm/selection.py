"""MCDM step: pick pi* from the evaluation matrix of feasible candidate plans (paper Sec. III-B/D).

The LLM (or mock) only *chooses*; feasibility comes from the deterministic validator, metric values from the
metric evaluation, and every answer is validated before use. The justification is kept separate from the evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..llm.base import LLMError, LLMProvider, LLMRequest
from ..redact import redact
from ..reference_metrics import DIRECTION
from .prompts import MCDM_JSON_SCHEMA, SELECT_SYSTEM, build_select_user
from .validation import SelectionValidation, validate_selection


@dataclass
class Candidate:
    plan_id: str
    metrics: dict[str, float]
    feasible: bool = True

    def as_dict(self) -> dict:
        return {"plan_id": self.plan_id, "metrics": dict(self.metrics), "feasible": self.feasible}


@dataclass
class MCDMResult:
    status: str  # "Success" | "Failure"
    selected_plan_id: Optional[str]
    justification: str  # LLM-generated text, NOT verified fact
    criteria_used: list[str]
    validation: Optional[SelectionValidation]
    preference: str
    n_candidates: int
    n_infeasible_excluded: int
    metric_ids: list[str]
    provider: str
    model: str
    is_mock: bool
    attempts: int
    prompt_chars: int
    latency_s: float
    meta: dict = field(default_factory=dict)
    error: Optional[str] = None
    warning_to_supervisor: Optional[str] = None
    attempt_log: list[dict] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == "Success"

    def to_dict(self) -> dict:
        return {
            "status": self.status, "selected_plan_id": self.selected_plan_id, "llm_explanation": self.justification,
            "criteria_used": self.criteria_used, "validation": self.validation.to_dict() if self.validation else None,
            "preference": self.preference, "n_candidates": self.n_candidates, "n_infeasible_excluded": self.n_infeasible_excluded,
            "metric_ids": self.metric_ids, "provider": self.provider, "model": self.model, "is_mock": self.is_mock,
            "attempts": self.attempts, "prompt_chars": self.prompt_chars, "latency_s": self.latency_s, "meta": self.meta,
            "error": self.error, "warning_to_supervisor": self.warning_to_supervisor, "attempt_log": self.attempt_log,
        }


def run_mcdm(candidates: list[Candidate], preference: str, provider: LLMProvider, metric_ids: Optional[list[str]] = None, max_attempts: int = 2, extra_descriptions: Optional[dict[str, str]] = None) -> MCDMResult:
    """Select pi*. Infeasible candidates are excluded *before* the model sees the matrix."""
    feasible = [c for c in candidates if c.feasible]
    excluded = len(candidates) - len(feasible)
    ids = metric_ids or sorted({m for c in feasible for m in c.metrics}, key=lambda s: (s[0] != "z", int(s[1:])))
    base = dict(preference=preference, n_candidates=len(feasible), n_infeasible_excluded=excluded, metric_ids=ids,
                provider=provider.name, model=provider.model, is_mock=provider.is_mock)
    if not feasible:
        return MCDMResult("Failure", None, "", [], None, attempts=0, prompt_chars=0, latency_s=0.0,
                          error="no feasible candidate plans to choose from",
                          warning_to_supervisor="No feasible plan is available; relax the constraints or revise the instance.", **base)
    if not preference.strip():
        return MCDMResult("Failure", None, "", [], None, attempts=0, prompt_chars=0, latency_s=0.0, error="empty preference text",
                          warning_to_supervisor="Please provide a natural-language trade-off preference.", **base)
    cdicts = [c.as_dict() for c in feasible]
    err_text = ""
    log: list[dict] = []
    t_all = time.perf_counter()
    last_meta: dict = {}
    prompt_chars = 0
    for attempt in range(1, max_attempts + 1):
        user = build_select_user(preference, cdicts, ids, err_text, extra_descriptions)
        prompt_chars = len(SELECT_SYSTEM) + len(user)
        req = LLMRequest(
            task="mcdm_select", system=SELECT_SYSTEM, user=user, json_schema=MCDM_JSON_SCHEMA, schema_name="mcdm_decision",
            payload={"preference": preference, "metric_ids": ids, "candidates": cdicts, "directions": {m: DIRECTION.get(m, "max") for m in ids}},
        )
        try:
            resp = provider.generate_structured(req)
        except LLMError as exc:
            err_text = redact(exc)
            log.append({"attempt": attempt, "outcome": "provider_error", "error": err_text})
            if type(exc).__name__ in ("LLMAuthError", "LLMRequestError", "ModelNotAvailableError"):
                break
            continue
        last_meta = resp.meta()
        val = validate_selection(resp.data, cdicts, ids)
        log.append({"attempt": attempt, "outcome": "valid" if val.ok else "invalid", "errors": val.errors, "response_meta": last_meta})
        if val.ok:
            return MCDMResult(
                "Success", resp.data["selected_plan_id"], str(resp.data["justification"]), list(resp.data.get("criteria_used", [])), val,
                attempts=attempt, prompt_chars=prompt_chars, latency_s=time.perf_counter() - t_all, meta=last_meta, attempt_log=log, **base,
            )
        err_text = "; ".join(val.errors)
    return MCDMResult(
        "Failure", None, "", [], None, attempts=len(log), prompt_chars=prompt_chars, latency_s=time.perf_counter() - t_all, meta=last_meta,
        error=err_text or "selection failed", attempt_log=log,
        warning_to_supervisor="MCDM could not obtain a valid selection. Check the preference text and the provider status. Last error: " + err_text,
        **base,
    )
