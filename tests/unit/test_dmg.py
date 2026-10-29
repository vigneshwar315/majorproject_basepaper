import json

from paper_replication.dmg.examples import expected_for, get_examples
from paper_replication.dmg.generation import DMGPrompt, run_dmg
from paper_replication.dmg.reference_specs import EXPRESSIONS
from paper_replication.llm.base import LLMAuthError, LLMRateLimitError
from paper_replication.llm.mock_provider import MockProvider
from tests.helpers import MATCH, Z1_SPEC, ScriptedProvider


def _prompt(mid="z1", oracle=True):
    exs = get_examples()
    return DMGPrompt("accepted over received", [e.view for e in exs], expected_for(mid, exs) if oracle else None)


def _gen(spec_json):
    return {"spec_json": spec_json}


WRONG_SPEC = json.dumps({"name": "w", "description": "", "expr": {"op": "const", "value": 3}})


def test_success_first_attempt():
    p = ScriptedProvider({"dmg_generate": [_gen(Z1_SPEC)], "dmg_validate": [MATCH]})
    r = run_dmg(_prompt(), p, 5)
    assert r.success and r.attempts_used == 1 and r.spec is not None
    assert r.trace[0].outcome == "success"
    assert [c.task for c in p.calls] == ["dmg_generate", "dmg_validate"]


def test_retry_after_static_error_feeds_error_back():
    p = ScriptedProvider({"dmg_generate": [_gen("{not json"), _gen(Z1_SPEC)], "dmg_validate": [MATCH]})
    r = run_dmg(_prompt(), p, 5)
    assert r.success and r.attempts_used == 2
    assert r.trace[0].outcome == "static_invalid"
    second = p.calls[1]
    assert "not valid JSON" in second.user and "{not json" in second.user


def test_retry_after_wrong_values():
    p = ScriptedProvider({"dmg_generate": [_gen(WRONG_SPEC), _gen(Z1_SPEC)], "dmg_validate": [MATCH]})
    r = run_dmg(_prompt(), p, 5)
    assert r.success and r.attempts_used == 2 and r.trace[0].outcome == "execution_failed"
    assert r.trace[0].execution_values == [3.0, 3.0, 3.0]


def test_llm_mismatch_triggers_retry():
    p = ScriptedProvider({"dmg_generate": [_gen(Z1_SPEC)], "dmg_validate": [{"verdict": "mismatch", "reason": "wrong"}, MATCH]})
    r = run_dmg(_prompt(), p, 5)
    assert r.success and r.attempts_used == 2 and r.trace[0].outcome == "validation_failed"


def test_failure_after_max_attempts_issues_supervisor_warning():
    p = ScriptedProvider({"dmg_generate": [_gen(WRONG_SPEC)], "dmg_validate": [MATCH]})
    r = run_dmg(_prompt(), p, 3)
    assert not r.success and r.status == "Failure" and r.attempts_used == 3 and len(r.trace) == 3
    assert r.spec is None and r.warning_to_supervisor and "clarify" in r.warning_to_supervisor
    assert sum(1 for c in p.calls if c.task == "dmg_generate") == 3


def test_fatal_error_aborts_immediately():
    p = ScriptedProvider({"dmg_generate": [LLMAuthError("bad key")]})
    r = run_dmg(_prompt(), p, 5)
    assert not r.success and r.attempts_used == 1 and "aborted" in r.warning_to_supervisor


def test_transient_provider_error_consumes_an_attempt_and_continues():
    p = ScriptedProvider({"dmg_generate": [LLMRateLimitError("slow down"), _gen(Z1_SPEC)], "dmg_validate": [MATCH]})
    r = run_dmg(_prompt(), p, 5)
    assert r.success and r.attempts_used == 2 and r.trace[0].outcome == "generation_error"


def test_execution_only_when_no_oracle():
    p = ScriptedProvider({"dmg_generate": [_gen(WRONG_SPEC)], "dmg_validate": [MATCH]})
    r = run_dmg(_prompt(oracle=False), p, 5)
    assert r.success  # weaker test: only checks that the metric runs and is finite


def test_max_attempts_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        run_dmg(_prompt(), MockProvider(), 0)


def test_mock_provider_generates_reference_metric_for_clear_prompt():
    from paper_replication.reference_metrics import DESCRIPTIONS

    exs = get_examples()
    r = run_dmg(DMGPrompt(DESCRIPTIONS["z3"], [e.view for e in exs], expected_for("z3", exs)), MockProvider(), 5)
    assert r.success and r.is_mock
    assert r.spec.expr == EXPRESSIONS["z3"]
