"""Google Gemini adapter (official ``google-genai`` SDK).

NOTE: this adapter could not be exercised against the live API from the build environment (network blocked);
it is covered by unit tests with a fake client. Verify your model id with ``python -m paper_replication.cli list-models``.
"""

from __future__ import annotations

import time

from .base import (
    LLMAuthError,
    LLMOutputError,
    LLMProvider,
    LLMRateLimitError,
    LLMRequest,
    LLMRequestError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    parse_json_object,
)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str, api_key: str, timeout_s: float = 60.0, max_retries: int = 3, temperature=None, client=None, **kw):
        super().__init__(model, max_retries, **kw)
        self._timeout_s = timeout_s
        self._temperature = temperature
        self._client = client
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=self._api_key, http_options=types.HttpOptions(timeout=int(self._timeout_s * 1000)))
        return self._client

    @staticmethod
    def _map_error(exc: Exception) -> Exception:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        msg = f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"
        name = type(exc).__name__.lower()
        if code in (401, 403):
            return LLMAuthError("authentication/permission error from Gemini (check GEMINI_API_KEY)")
        if code == 429:
            return LLMRateLimitError("Gemini rate limit / quota exceeded")
        if code in (400, 404):
            return LLMRequestError(f"Gemini rejected the request ({code}): {msg}")
        if code is not None and 500 <= int(code) < 600:
            return LLMUnavailableError(f"Gemini server error ({code})")
        if "timeout" in name or "timed out" in msg.lower():
            return LLMTimeoutError("Gemini request timed out")
        if "connect" in name or "network" in msg.lower():
            return LLMUnavailableError("could not reach Gemini")
        return LLMUnavailableError(f"unexpected Gemini error: {msg}")

    def _call(self, req: LLMRequest) -> LLMResponse:
        from google.genai import types

        client = self._get_client()
        cfg = {
            "system_instruction": req.system,
            "response_mime_type": "application/json",
            "response_json_schema": req.json_schema,
        }
        if self._temperature is not None:
            cfg["temperature"] = self._temperature
        t0 = time.perf_counter()
        try:
            resp = client.models.generate_content(model=self.model, contents=req.user, config=types.GenerateContentConfig(**cfg))
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors
            raise self._map_error(exc) from None
        latency = time.perf_counter() - t0
        text = getattr(resp, "text", None)
        if not text:
            raise LLMOutputError("Gemini returned an empty response (possibly blocked or truncated)")
        data = parse_json_object(text)
        um = getattr(resp, "usage_metadata", None)
        in_t = getattr(um, "prompt_token_count", None) if um else None
        out_t = getattr(um, "response_token_count", None) if um else None
        if out_t is None and um:
            out_t = getattr(um, "candidates_token_count", None)
        return LLMResponse(data, text, self.name, self.model, False, latency, in_t, out_t)

    def list_models(self) -> list[str]:
        try:
            return [m.name for m in self._get_client().models.list()]
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from None
