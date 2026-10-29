"""OpenAI adapter (official ``openai`` SDK, Chat Completions with JSON-schema structured output).

NOTE: not exercised against the live API from the build environment (network blocked); covered by unit tests
with a fake client. Verify your model id with ``python -m paper_replication.cli list-models``.
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
    strict_schema,
)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str, api_key: str, timeout_s: float = 60.0, max_retries: int = 3, temperature=None, client=None, **kw):
        super().__init__(model, max_retries, **kw)
        self._timeout_s = timeout_s
        self._temperature = temperature
        self._client = client
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            # SDK-level retries are disabled: retry policy is ours and visible in the response metadata.
            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout_s, max_retries=0)
        return self._client

    @staticmethod
    def _map_error(exc: Exception) -> Exception:
        name = type(exc).__name__
        msg = str(exc)
        if name in ("AuthenticationError", "PermissionDeniedError"):
            return LLMAuthError("authentication/permission error from OpenAI (check OPENAI_API_KEY)")
        if name == "RateLimitError":
            return LLMRateLimitError("OpenAI rate limit / quota exceeded")
        if name == "APITimeoutError":
            return LLMTimeoutError("OpenAI request timed out")
        if name in ("APIConnectionError", "InternalServerError"):
            return LLMUnavailableError(f"OpenAI unavailable ({name})")
        if name in ("BadRequestError", "NotFoundError", "UnprocessableEntityError"):
            return LLMRequestError(f"OpenAI rejected the request ({name}): {msg}")
        return LLMUnavailableError(f"unexpected OpenAI error ({name}): {msg}")

    def _call(self, req: LLMRequest) -> LLMResponse:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": req.schema_name, "strict": True, "schema": strict_schema(req.json_schema)}},
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors
            raise self._map_error(exc) from None
        latency = time.perf_counter() - t0
        try:
            choice = resp.choices[0]
            text = choice.message.content
            if getattr(choice.message, "refusal", None):
                raise LLMOutputError("OpenAI model refused the request")
        except (AttributeError, IndexError) as exc:
            raise LLMOutputError("unexpected OpenAI response structure") from exc
        if not text:
            raise LLMOutputError("OpenAI returned an empty response")
        data = parse_json_object(text)
        usage = getattr(resp, "usage", None)
        return LLMResponse(data, text, self.name, self.model, False, latency,
                           getattr(usage, "prompt_tokens", None) if usage else None,
                           getattr(usage, "completion_tokens", None) if usage else None)

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self._get_client().models.list()]
        except Exception as exc:  # noqa: BLE001
            raise self._map_error(exc) from None
