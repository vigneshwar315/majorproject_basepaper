import pytest

from paper_replication.experiments.prompt_sets import (
    TRADEOFF_SEEDS,
    degrade_metric_text,
    degraded_metric_prompt_set,
    degraded_tradeoff_prompt_set,
    metric_prompt_set,
    tradeoff_prompt_set,
)
from paper_replication.mcdm.reference_rules import RULES


@pytest.mark.parametrize("ptype", ["T", "TF", "TFP"])
def test_metric_prompt_sets_have_thirty_unique_prompts(ptype):
    ps = metric_prompt_set(ptype)
    assert len(ps) == 30 and len({p.text for p in ps}) == 30
    assert {p.metric_id for p in ps} == {f"z{i}" for i in range(1, 11)}
    if ptype != "T":
        assert all("$" in p.text or "\\" in p.text for p in ps)


def test_prompt_type_must_be_valid():
    with pytest.raises(ValueError):
        metric_prompt_set("X")


def test_table4_numbering_gap_kept():
    assert sorted(TRADEOFF_SEEDS) == [1, 2, 3, 5, 6, 7, 8, 9, 10]  # the paper has no prompt #4 (uncertainty U16)
    assert set(TRADEOFF_SEEDS) == set(RULES)


def test_tradeoff_prompt_set_size_and_sources():
    ps = tradeoff_prompt_set()
    assert len(ps) == 30
    assert {p.source for p in ps} == {"table4_seed", "handwritten_paraphrase"}


@pytest.mark.parametrize("level", ["informal", "unclear", "ambiguous"])
def test_degradations_change_text_and_are_labelled(level):
    ps = degraded_metric_prompt_set("T", level)
    assert len(ps) == 30
    base = {p.metric_id + p.text for p in metric_prompt_set("T")}
    assert sum(1 for p in ps if p.metric_id + p.text not in base) >= 20
    assert all(p.source == f"rule_degraded:{level}" for p in ps)


def test_formulas_are_not_degraded_for_tf():
    tf = metric_prompt_set("TF")[0]
    degraded = degraded_metric_prompt_set("TF", "informal")[0]
    formula = tf.text.split("\n", 1)[1]
    assert formula in degraded.text


def test_tradeoff_degradation_levels():
    for level in (1, 2, 3):
        ps = degraded_tradeoff_prompt_set(level)
        assert len(ps) == 30 and all(p.degradation_level for p in ps)
    assert degrade_metric_text("Introduce a new metric", "informal")
