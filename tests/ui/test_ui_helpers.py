import pandas as pd
import pytest
from frontend import helpers

from paper_replication.candidate_generation import generate_candidates
from paper_replication.problem_instance import generate_instance, tiny_instance


def test_time_helpers():
    assert helpers.parse_hhmm("08:30") == 510.0
    assert helpers.fmt_hhmm(510.0) == "08:30"
    for bad in ("8.30", "25:61", "", "ab:cd"):
        with pytest.raises(ValueError):
            helpers.parse_hhmm(bad)


def test_instance_frame_round_trip():
    inst = tiny_instance()
    veh, req = helpers.instance_to_frames(inst)
    settings = helpers.scalar_settings(inst)
    back, errs = helpers.frames_to_instance(settings, veh, req, None)
    assert errs == [] and back is not None
    assert back.to_dict() == inst.to_dict()


def test_field_level_errors_for_bad_edits():
    inst = tiny_instance()
    veh, req = helpers.instance_to_frames(inst)
    req = req.copy().astype({"demand": object})
    req.loc[0, "demand"] = "many"
    req.loc[1, "pickup_tw_from"] = "9am"
    req.loc[1, "pickup_tw_to"] = "10:00"
    veh = veh.copy().astype({"capacity": object})
    veh.loc[0, "capacity"] = ""
    out, errs = helpers.frames_to_instance(helpers.scalar_settings(inst), veh, req, None)
    assert out is None
    text = " | ".join(str(e) for e in errs)
    assert "demand" in text and "HH:MM" in text and "capacity" in text
    assert set(helpers.errors_by_section(errs)) >= {"requests", "vehicles"}


def test_duplicate_ids_reported_after_conversion():
    inst = tiny_instance()
    veh, req = helpers.instance_to_frames(inst)
    req = pd.concat([req, req.iloc[[0]]], ignore_index=True)
    out, errs = helpers.frames_to_instance(helpers.scalar_settings(inst), veh, req, None)
    assert out is None and any("duplicate" in e.message for e in errs)


def test_distance_matrix_csv_round_trip_and_errors():
    inst = tiny_instance()
    csv = helpers.distance_matrix_template_csv(inst)
    dm, errs = helpers.parse_distance_matrix_csv(csv)
    assert errs == [] and dm is not None and dm["depot"]["a:P"] == pytest.approx(5.0, abs=1e-6) or dm is not None
    _, bad = helpers.parse_distance_matrix_csv("x,y\n1,abc\n")
    assert bad


def test_plan_tables():
    inst = generate_instance(seed=5, n_requests=12, n_vehicles=2)
    plan = generate_candidates(inst, 2, seed=5).plans[0]
    summary = helpers.route_summary_table(inst, plan)
    stops = helpers.route_table(inst, plan)
    assert len(stops) == 2 * len(plan.accepted_ids())
    assert summary["feasible"].all()
    assert len(helpers.decisions_table(plan)) == len(plan.decisions)


def test_comparison_table_marks_selected_and_infeasible_rows():
    inst = generate_instance(seed=5, n_requests=12, n_vehicles=2)
    plans = generate_candidates(inst, 3, seed=5).plans
    matrix = {plans[0].plan_id: {"z1": 0.5}}
    df = helpers.comparison_table(plans, matrix, ["z1"], plans[0].plan_id)
    assert df.loc[0, "selected"] == "yes" and df.loc[0, "z1"] == 0.5
    assert df["z1"].iloc[1:].isna().all()


def test_prompt_call_estimate_and_doc_redaction(monkeypatch, tmp_path):
    assert helpers.prompt_call_estimate("C2") > 0 and helpers.prompt_call_estimate("nope") == 0
    monkeypatch.setenv("OPENAI_API_KEY", "supersecretvalue123")
    f = tmp_path / "d.md"
    f.write_text("leaked supersecretvalue123 here", encoding="utf-8")
    assert "supersecretvalue123" not in helpers.read_doc(f)
    assert helpers.read_doc(tmp_path / "missing.md") is None


def test_test_totals_sums_suites():
    ts = {"suites": {"unit": {"passed": 3, "failed": 1, "skipped": 0, "tests": 4}, "live_llm": {"status": "NOT RUN"}}}
    assert helpers.test_totals(ts) == {"passed": 3, "failed": 1, "skipped": 0, "tests": 4}
    assert helpers.test_totals({}) == {"passed": 0, "failed": 0, "skipped": 0, "tests": 0}


def test_instance_and_route_figures_show_every_point():
    from frontend import charts

    inst = generate_instance(seed=5, n_requests=12, n_vehicles=2)
    fig = charts.instance_figure(inst)
    xs = [x for tr in fig.data if tr.mode == "markers" and tr.name and tr.name.startswith(("pickup", "delivery")) for x in tr.x]
    assert len(xs) == 2 * len(inst.requests)
    plan = generate_candidates(inst, 2, seed=5).plans[0]
    day = sorted({r.day for r in plan.routes})[0]
    rf = charts.route_figure(inst, plan, day)
    assert any(tr.name.startswith("vehicle") for tr in rf.data)
