"""Provider-independent LLM interface used by DMG and MCDM.

Requirements honoured here (project brief Sec. 2/7): configurable adapter, no silent provider/model fallback,
typed errors, bounded retries for transient failures, no secrets in exceptions.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..redact import redact


class LLMError(RuntimeError):
    """Base class. Messages are always passed through ``redact``."""

    def __init__(self, message: str):
        super().__init__(redact(message))


class LLMAuthError(LLMError): ...
class LLMRateLimitError(LLMError): ...
class LLMTimeoutError(LLMError): ...
class LLMOutputError(LLMError): ...  # malformed / non-JSON / schema-violating output
class LLMUnavailableError(LLMError): ...  # provider outage / connection problems
class LLMRequestError(LLMError): ...  # rejected request (bad model id, unsupported parameter...)
class ModelNotAvailableError(LLMError): ...  # configured model not offered by the provider


TRANSIENT = (LLMRateLimitError, LLMTimeoutError, LLMUnavailableError)


@dataclass
class LLMRequest:
    task: str  # dmg_generate | dmg_validate | mcdm_select | paraphrase | ...
    system: str
    user: str
    json_schema: dict
    schema_name: str = "response"
    payload: dict = field(default_factory=dict)  # structured copy of the inputs; used ONLY by the mock provider


@dataclass
class LLMResponse:
    data: dict
    raw_text: str
    provider: str
    model: str
    is_mock: bool
    latency_s: float
    input_tokens: Optional[int] = None  # None = unavailable (never 0 as a stand-in)
    output_tokens: Optional[int] = None
    attempts: int = 1

    def meta(self) -> dict:
        return {
            "provider": self.provider, "model": self.model, "is_mock": self.is_mock,
            "latency_s": round(self.latency_s, 4), "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "attempts": self.attempts,
        }


class LLMProvider(ABC):
    name = "base"
    is_mock = False

    def __init__(self, model: str, max_retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        self.model = model
        self.max_retries = max(0, int(max_retries))
        self._sleep = sleep

    @abstractmethod
    def _call(self, req: LLMRequest) -> LLMResponse: ...

    def list_models(self) -> list[str]:  # pragma: no cover - network
        raise NotImplementedError

    def verify_model(self) -> None:
        """Raise ``ModelNotAvailableError`` if the configured model is not offered to this key."""
        available = self.list_models()
        norm = {m.split("/", 1)[-1] if m.startswith("models/") else m for m in available}
        want = self.model.split("/", 1)[-1] if self.model.startswith("models/") else self.model
        if want not in norm:
            sample = ", ".join(sorted(norm)[:15])
            raise ModelNotAvailableError(f"model {self.model!r} is not offered by provider {self.name!r}. Examples of available ids: {sample}")

    def generate_structured(self, req: LLMRequest) -> LLMResponse:
        """Call the provider with bounded retries for transient errors only. Never switches model/provider."""
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._call(req)
                resp.attempts = attempt
                if not isinstance(resp.data, dict):
                    raise LLMOutputError("provider returned a non-object JSON value")
                return resp
            except TRANSIENT as exc:
                if attempt > self.max_retries:
                    raise
                self._sleep(min(30.0, 2.0 ** (attempt - 1)))
                _ = exc

    def info(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "is_mock": self.is_mock}


def parse_json_object(text: str) -> dict:
    """Parse model output as a JSON object, tolerating code fences. Raises ``LLMOutputError``."""
    import json
    import re

    t = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", t, re.DOTALL)
    if m:
        t = m.group(1)
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise LLMOutputError("model output must be a JSON object")
    return obj


def strict_schema(schema: dict) -> dict:
    """Return a copy of a simple object schema made strict (all props required, no additional props)."""
    import copy

    s = copy.deepcopy(schema)

    def walk(n):
        if isinstance(n, dict):
            if n.get("type") == "object" and "properties" in n:
                n["required"] = list(n["properties"].keys())
                n["additionalProperties"] = False
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(s)
    return s
