
from paper_replication.experiments.paper_reference import table9_matrix
from paper_replication.llm.base import LLMAuthError
from paper_replication.llm.mock_provider import MockProvider
from paper_replication.mcdm.reference_rules import RULES, acceptable_plans, is_correct, regret
from paper_replication.mcdm.selection import Candidate, run_mcdm
from paper_replication.mcdm.validation import check_grounding, validate_selection
from tests.helpers import ScriptedProvider

CANDS = [Candidate("pi_1", {"z1": 0.9, "z3": 100.0}), Candidate("pi_2", {"z1": 0.8, "z3": 60.0}), Candidate("pi_3", {"z1": 1.0, "z3": 500.0}, feasible=False)]
IDS = ["z1", "z3"]


def _ok(plan="pi_1", text="Plan chosen because z1 = 0.9 is highest among feasible plans.", crit=("z1",)):
    return {"selected_plan_id": plan, "justification": text, "criteria_used": list(crit)}


def test_valid_selection():
    p = ScriptedProvider({"mcdm_select": [_ok()]})
    r = run_mcdm(CANDS, "maximise acceptance", p, IDS)
    assert r.success and r.selected_plan_id == "pi_1" and r.validation.ok
    assert r.validation.claims_checked == 1 and r.validation.claims_supported == 1


def test_infeasible_plans_never_reach_the_model():
    p = ScriptedProvider({"mcdm_select": [_ok()]})
    r = run_mcdm(CANDS, "maximise acceptance", p, IDS)
    assert r.n_infeasible_excluded == 1 and r.n_candidates == 2
    assert "pi_3" not in p.calls[0].user and all(c["plan_id"] != "pi_3" for c in p.calls[0].payload["candidates"])


def test_invented_plan_id_rejected_then_retried():
    p = ScriptedProvider({"mcdm_select": [_ok("pi_99"), _ok("pi_2", "z3 = 60 is shortest.", ("z3",))]})
    r = run_mcdm(CANDS, "short tours", p, IDS)
    assert r.success and r.selected_plan_id == "pi_2" and r.attempts == 2
    assert "invented id" in p.calls[1].user


def test_choosing_infeasible_plan_is_rejected():
    v = validate_selection(_ok("pi_3"), [c.as_dict() for c in CANDS], IDS)
    assert not v.ok and any("feasibility" in e for e in v.errors)


def test_persistent_invalid_output_fails_with_warning():
    p = ScriptedProvider({"mcdm_select": [_ok("nope")]})
    r = run_mcdm(CANDS, "x", p, IDS, max_attempts=2)
    assert not r.success and r.selected_plan_id is None and r.warning_to_supervisor and r.attempts == 2


def test_unknown_criteria_and_empty_justification():
    cands = [c.as_dict() for c in CANDS]
    assert not validate_selection(_ok(crit=("z77",)), cands, IDS).ok
    assert not validate_selection(_ok(text="  "), cands, IDS).ok


def test_grounding_flags_unsupported_numbers():
    v = check_grounding("z1 = 0.55 and z3 = 60", "pi_2", [c.as_dict() for c in CANDS], IDS)
    assert v.claims_checked == 2 and v.claims_supported == 1 and v.unsupported_claims == ["z1 = 0.55"]


def test_no_feasible_candidates_and_empty_preference():
    p = ScriptedProvider({"mcdm_select": [_ok()]})
    assert not run_mcdm([Candidate("a", {"z1": 1.0}, feasible=False)], "x", p).success
    assert not run_mcdm(CANDS, "   ", p, IDS).success
    assert p.calls == []


def test_fatal_provider_error_stops_retrying():
    p = ScriptedProvider({"mcdm_select": [LLMAuthError("bad")]})
    r = run_mcdm(CANDS, "x", p, IDS, max_attempts=3)
    assert not r.success and len(p.calls) == 1


def test_reference_rules_on_table9_matrix():
    ids, matrix = table9_matrix()
    assert len(matrix) == 10 and "z1" in ids
    usable = [k for k, stages in RULES.items() if all(st.metric in ids for st in stages)]
    assert usable == [1, 2, 5, 6, 7]  # rules 3, 8, 9, 10 need extension metrics z11-z14, absent from Table 9
    for rule in usable:
        acc = acceptable_plans(rule, matrix)
        assert acc, rule
        assert all(is_correct(rule, matrix, a) for a in acc)
    some = next(iter(matrix))
    assert set(regret(1, matrix, some)) == {"z1", "z2"}
    assert not is_correct(1, matrix, None)


def test_mock_provider_selects_a_feasible_plan_for_each_rule_prompt():
    ids, matrix = table9_matrix()
    from paper_replication.experiments.prompt_sets import TRADEOFF_SEEDS

    for text in TRADEOFF_SEEDS.values():
        r = run_mcdm([Candidate(p, v) for p, v in matrix.items()], text, MockProvider(), ids)
        assert r.success and r.selected_plan_id in matrix
        assert "[MOCK RATIONALE" in r.justification
