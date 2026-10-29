"""Deterministic MOCK provider for offline tests and demonstrations.

IMPORTANT: this is NOT an LLM. It simulates one with transparent keyword rules so that the whole pipeline (DMG loop,
validation, MCDM, experiments, UI) can run without network access or credentials. Everything it produces is tagged
``is_mock=True`` and must never be presented as a real LLM result. Results obtained with it measure the pipeline,
not any language model.
"""

from __future__ import annotations

import json
import re
import time

from ..dmg.metric_specification import parse_spec
from ..dmg.reference_specs import EXPRESSIONS, KEYWORDS, METRIC_NAMES
from ..reference_metrics import DIRECTION
from .base import LLMOutputError, LLMProvider, LLMRequest, LLMResponse

MOCK_MODEL = "mock-deterministic"
_ORDER = list(EXPRESSIONS)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def rank_metrics(text: str) -> list[tuple[str, float]]:
    """Metric ids ranked by keyword evidence (descending score, ties by id order)."""
    t = _norm(text)
    scored = []
    for idx, mid in enumerate(_ORDER):
        s = sum(w for phrase, w in KEYWORDS[mid] if phrase in t)
        scored.append((mid, s, idx))
    scored.sort(key=lambda x: (-x[1], x[2]))
    return [(m, s) for m, s, _ in scored]


_GENERIC_WRONG = {"op": "agg", "fn": "count", "over": "requests"}  # "number of requests" - plausible but wrong


def _mock_generate(payload: dict) -> dict:
    ranked = rank_metrics(payload.get("metric", ""))
    attempt = int(payload.get("attempt", 0))
    if not ranked or ranked[0][1] <= 0:
        expr = _GENERIC_WRONG
        name = "count_requests"
    else:
        pick = ranked[min(attempt, len(ranked) - 1)][0]
        expr, name = EXPRESSIONS[pick], METRIC_NAMES[pick]
    spec = {"name": name, "description": (payload.get("metric") or "")[:200], "expr": expr}
    return {"spec_json": json.dumps(spec, sort_keys=True)}


def _mock_validate(payload: dict) -> dict:
    ranked = rank_metrics(payload.get("metric", ""))
    try:
        spec = parse_spec(payload.get("spec_json", ""))
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "mismatch", "reason": f"specification could not be parsed: {exc}"}
    if not ranked or ranked[0][1] <= 0:
        return {"verdict": "mismatch", "reason": "the description does not match any metric the mock understands"}
    best = ranked[0][0]
    want = json.dumps(EXPRESSIONS[best], sort_keys=True, separators=(",", ":"))
    if spec.fingerprint() == want:
        return {"verdict": "match", "reason": f"specification is consistent with the description (mock keyword check: {best})"}
    return {"verdict": "mismatch", "reason": "specification does not appear to compute the quantity described (mock keyword check)"}


# ---- MCDM -------------------------------------------------------------------------------------------------
_RULES: list[tuple[tuple[str, ...], dict[str, float]]] = [
    (("as soon as possible", "today rather than tomorrow", "same day", "today"), {"z2": 2.0, "z9": 1.0}),
    (("tight time-window", "tight time window", "tight window", "exact time"), {"z11": 3.0}),
    (("zip", "peak hours", "congestion"), {"z8": 2.0}),
    (("balance", "underutiliz", "overloading", "workload"), {"z6": 2.0}),
    (("reserve a portion", "high-priority", "high priority"), {"z10": 2.0}),
    (("returning to the depot", "after a specified time", "overtime"), {"z12": 2.0}),
    (("consecutive days", "long routes", "fatigue"), {"z13": 2.0}),
    (("familiarity", "different drivers", "customer familiarity"), {"z14": 2.0}),
]


def mock_weights(preference: str) -> dict[str, float]:
    t = _norm(preference)
    w: dict[str, float] = {"z1": 3.0}
    for phrases, adds in _RULES:
        if any(p in t for p in phrases):
            for k, v in adds.items():
                w[k] = w.get(k, 0.0) + v
    longer_ok = "even if" in t or "even when" in t
    wants_short = ("does not result in longer" in t) or ("without longer" in t) or ("no longer" in t) or ("not result in longer" in t)
    if wants_short and not longer_ok:
        w["z3"] = w.get("z3", 0.0) + 2.0
        w["z4"] = w.get("z4", 0.0) + 1.0
    return w


def _mock_select(payload: dict) -> dict:
    cands = [c for c in payload.get("candidates", []) if c.get("feasible", True)]
    if not cands:
        raise LLMOutputError("mock: no feasible candidates supplied")
    metric_ids = [m for m in payload.get("metric_ids", []) if all(m in c["metrics"] for c in cands)]
    weights = {m: w for m, w in mock_weights(payload.get("preference", "")).items() if m in metric_ids}
    lo = {m: min(c["metrics"][m] for c in cands) for m in metric_ids}
    hi = {m: max(c["metrics"][m] for c in cands) for m in metric_ids}

    def score(c):
        total = 0.0
        for m, w in weights.items():
            span = hi[m] - lo[m]
            norm = 1.0 if span == 0 else (c["metrics"][m] - lo[m]) / span
            if DIRECTION.get(m, "max") == "min":
                norm = 1.0 - norm
            total += w * norm
        return total

    best = sorted(cands, key=lambda c: (-round(score(c), 9), -c["metrics"].get("z1", 0.0), c["metrics"].get("z3", 0.0), c["plan_id"]))[0]
    used = sorted(weights, key=lambda m: -weights[m])
    vals = ", ".join(f"{m} = {best['metrics'][m]:.4g}" for m in used[:5])
    text = (
        f"[MOCK RATIONALE - rule-based stand-in, not an LLM] Plan {best['plan_id']} is selected because it gives the best weighted "
        f"compromise on the criteria suggested by the preference ({vals}), computed over {len(cands)} feasible candidate plans."
    )
    return {"selected_plan_id": best["plan_id"], "justification": text, "criteria_used": used}


class MockProvider(LLMProvider):
    name = "mock"
    is_mock = True

    def __init__(self, model: str = MOCK_MODEL, max_retries: int = 0, **kw):
        super().__init__(model or MOCK_MODEL, max_retries, **kw)

    def list_models(self) -> list[str]:
        return [MOCK_MODEL]

    def _call(self, req: LLMRequest) -> LLMResponse:
        t0 = time.perf_counter()
        if req.task == "dmg_generate":
            data = _mock_generate(req.payload)
        elif req.task == "dmg_validate":
            data = _mock_validate(req.payload)
        elif req.task == "mcdm_select":
            data = _mock_select(req.payload)
        else:
            raise LLMOutputError(f"mock provider does not implement task {req.task!r}")
        text = json.dumps(data, sort_keys=True)
        return LLMResponse(data, text, self.name, self.model, True, time.perf_counter() - t0, None, None)
