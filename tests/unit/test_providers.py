"""Provider adapters against fake SDK clients. NO live API call is made here (see report: live path BLOCKED)."""

from types import SimpleNamespace

import pytest

from paper_replication.llm.base import (
    LLMAuthError,
    LLMOutputError,
    LLMRateLimitError,
    LLMRequest,
    LLMTimeoutError,
    LLMUnavailableError,
    ModelNotAvailableError,
    parse_json_object,
    strict_schema,
)
from paper_replication.llm.gemini_provider import GeminiProvider
from paper_replication.llm.mock_provider import MockProvider
from paper_replication.llm.openai_provider import OpenAIProvider

REQ = LLMRequest("mcdm_select", "sys", "user", {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}, "x")


class RateLimitError(Exception): ...
class AuthenticationError(Exception): ...
class APITimeoutError(Exception): ...


# ---------------- OpenAI ----------------
def _oa_resp(text, prompt=11, completion=7, refusal=None):
    msg = SimpleNamespace(content=text, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


class FakeOpenAI:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.kwargs = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: [SimpleNamespace(id="m-1"), SimpleNamespace(id="m-2")])

    def _create(self, **kw):
        self.kwargs.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def test_openai_success_tokens_and_strict_schema():
    c = FakeOpenAI([_oa_resp('{"a": "x"}')])
    p = OpenAIProvider("m-1", "k", client=c)
    r = p.generate_structured(REQ)
    assert r.data == {"a": "x"} and (r.input_tokens, r.output_tokens) == (11, 7) and not r.is_mock
    sent = c.kwargs[0]
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert "temperature" not in sent  # not sent unless configured


def test_openai_temperature_only_when_set():
    c = FakeOpenAI([_oa_resp('{"a": "x"}')])
    OpenAIProvider("m-1", "k", temperature=0.0, client=c).generate_structured(REQ)
    assert c.kwargs[0]["temperature"] == 0.0


def test_openai_error_mapping_and_bounded_retry():
    sleeps = []
    c = FakeOpenAI([RateLimitError("429"), RateLimitError("429"), _oa_resp('{"a": "x"}')])
    p = OpenAIProvider("m-1", "k", client=c, max_retries=3, sleep=sleeps.append)
    r = p.generate_structured(REQ)
    assert r.attempts == 3 and sleeps == [1.0, 2.0]
    c2 = FakeOpenAI([RateLimitError("429")] * 10)
    with pytest.raises(LLMRateLimitError):
        OpenAIProvider("m-1", "k", client=c2, max_retries=2, sleep=lambda s: None).generate_structured(REQ)
    assert len(c2.kwargs) == 3  # 1 try + 2 retries, then stop
    c3 = FakeOpenAI([AuthenticationError("bad key sk-SECRETSECRETSECRET123")])
    with pytest.raises(LLMAuthError) as ei:
        OpenAIProvider("m-1", "k", client=c3, max_retries=3).generate_structured(REQ)
    assert len(c3.kwargs) == 1 and "SECRETSECRET" not in str(ei.value)  # auth errors are not retried and do not leak
    with pytest.raises(LLMTimeoutError):
        OpenAIProvider("m-1", "k", client=FakeOpenAI([APITimeoutError("t")]), max_retries=0).generate_structured(REQ)


def test_openai_bad_outputs():
    for outcome in (_oa_resp("not json"), _oa_resp(""), _oa_resp("[1]"), _oa_resp(None, refusal="no")):
        with pytest.raises(LLMOutputError):
            OpenAIProvider("m-1", "k", client=FakeOpenAI([outcome]), max_retries=0).generate_structured(REQ)


def test_openai_verify_model_has_no_fallback():
    p = OpenAIProvider("m-1", "k", client=FakeOpenAI([]))
    p.verify_model()
    with pytest.raises(ModelNotAvailableError, match="m-3"):
        OpenAIProvider("m-3", "k", client=FakeOpenAI([])).verify_model()


# ---------------- Gemini ----------------
class GeminiAPIError(Exception):
    def __init__(self, code, message="x"):
        super().__init__(message)
        self.code, self.message = code, message


def _gm_resp(text, p=5, c=3):
    return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(prompt_token_count=p, response_token_count=c, candidates_token_count=c))


class FakeGemini:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._gen, list=lambda: [SimpleNamespace(name="models/g-1")])

    def _gen(self, **kw):
        self.calls.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


def test_gemini_success_and_schema_passthrough():
    c = FakeGemini([_gm_resp('```json\n{"a": "x"}\n```')])
    r = GeminiProvider("g-1", "k", client=c).generate_structured(REQ)
    assert r.data == {"a": "x"} and (r.input_tokens, r.output_tokens) == (5, 3)
    cfg = c.calls[0]["config"]
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema == REQ.json_schema
    assert c.calls[0]["model"] == "g-1"


def test_gemini_missing_usage_is_none_not_zero():
    r = GeminiProvider("g-1", "k", client=FakeGemini([SimpleNamespace(text='{"a": "x"}', usage_metadata=None)])).generate_structured(REQ)
    assert r.input_tokens is None and r.output_tokens is None


def test_gemini_error_mapping():
    cases = [(GeminiAPIError(401), LLMAuthError), (GeminiAPIError(429), LLMRateLimitError), (GeminiAPIError(503), LLMUnavailableError)]
    for exc, expected in cases:
        with pytest.raises(expected):
            GeminiProvider("g-1", "k", client=FakeGemini([exc]), max_retries=0).generate_structured(REQ)
    with pytest.raises(LLMOutputError):
        GeminiProvider("g-1", "k", client=FakeGemini([_gm_resp("")]), max_retries=0).generate_structured(REQ)


def test_gemini_verify_model_normalises_prefix():
    GeminiProvider("g-1", "k", client=FakeGemini([])).verify_model()
    GeminiProvider("models/g-1", "k", client=FakeGemini([])).verify_model()
    with pytest.raises(ModelNotAvailableError):
        GeminiProvider("g-9", "k", client=FakeGemini([])).verify_model()


# ---------------- shared helpers ----------------
def test_parse_json_object_and_strict_schema():
    assert parse_json_object('```\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMOutputError):
        parse_json_object("nope")
    s = strict_schema({"type": "object", "properties": {"x": {"type": "string"}, "y": {"type": "object", "properties": {"z": {"type": "number"}}}}})
    assert s["required"] == ["x", "y"] and s["properties"]["y"]["additionalProperties"] is False


def test_mock_is_marked_and_rejects_unknown_tasks():
    m = MockProvider()
    assert m.is_mock and m.info()["is_mock"] is True
    with pytest.raises(LLMOutputError):
        m.generate_structured(LLMRequest("unknown", "s", "u", {}, "x"))
