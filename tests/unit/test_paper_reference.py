import json

from paper_replication.config import DATA_DIR
from paper_replication.experiments.paper_reference import load_paper_reference, table9_matrix
from paper_replication.experiments.reporting import comparison_with_paper


def test_reference_values_loaded_with_page_citations():
    ref = load_paper_reference()
    for key in ("table3_C1_dmg_accuracy", "table5_C2_mcdm_accuracy", "table6_C3_robustness", "table7_C4_prompt_clarity", "table8_C5_tokens_time"):
        assert "page" in ref[key], key
    assert ref["table3_C1_dmg_accuracy"]["rows"]["T"] == {"generated": 27, "avg_attempts": 1.42, "tests_passed": 27}
    assert ref["table5_C2_mcdm_accuracy"]["correct_by_m"] == {"10": 30, "50": 28, "100": 25}


def test_table9_shape():
    ids, matrix = table9_matrix()
    assert ids == [f"z{i}" for i in range(1, 11)]
    assert len(matrix) == 10 and all(len(v) == 10 for v in matrix.values())


def test_comparison_refuses_mock_and_synthetic_runs():
    summary = {"per_type": {"T": {"generated": 30, "avg_attempts": 1.0, "tests_passed": 30}}}
    table, valid, reason = comparison_with_paper("C1", summary, "synthetic_demonstration")
    assert valid is False and "NOT compared" in reason
    _, valid2, reason2 = comparison_with_paper("C1", summary, "alternative_model")
    assert valid2 is True and "Indicative" in reason2
    _, valid3, _ = comparison_with_paper("C6", {}, "alternative_model")
    assert valid3 is False


def test_data_file_is_valid_json():
    json.loads((DATA_DIR / "paper_reference_values.json").read_text(encoding="utf-8"))
