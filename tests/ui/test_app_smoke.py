"""Headless smoke tests: every page renders against the real backend (mock provider) without raising."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "app.py")

PAGES = ["overview", "problem", "candidates", "dmg", "mcdm", "experiments", "logs_reports"]


def _page_script(page: str, with_result: bool) -> str:
    return f"""
import sys
sys.path[:0] = ["src", "."]
import streamlit as st
from frontend import state, theme
from frontend.pages import {page} as page
state.init_state()
theme.inject_css()
if {with_result!r} and st.session_state.get("result") is None:
    from paper_replication.llm.mock_provider import MockProvider
    from paper_replication.pipeline import MetricRequest, run_pipeline
    from paper_replication.problem_instance import generate_instance
    inst = generate_instance(seed=5, n_requests=14, n_vehicles=2)
    res = run_pipeline(inst, "Maximize accepted requests, then minimise total travel.", MockProvider(), m=6, seed=5, save=False,
                       metric_requests=[MetricRequest("Introduce a metric: the share of received requests that are accepted.", "z1")])
    st.session_state["result"] = res
    st.session_state["selected_plan"] = res.mcdm.selected_plan_id if res.mcdm else None
page.render()
"""


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_result(page):
    at = AppTest.from_string(_page_script(page, False), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", ["candidates", "dmg", "mcdm", "logs_reports", "overview"])
def test_page_renders_with_result(page):
    at = AppTest.from_string(_page_script(page, True), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]


def test_full_app_starts():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_one_click_demo_produces_results_and_map():
    at = AppTest.from_string(_page_script("problem", False), default_timeout=120).run()
    demo = [b for b in at.button if b.label == "Run full demo"]
    assert demo, "demo button missing"
    demo[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    res = at.session_state["result"]
    assert res is not None and res.candidate_set.produced_m >= 2
    assert res.dmg_results["d1"].success and "d1" in res.metric_ids
    assert at.session_state["preference"].startswith("Maximize")
    assert any("Demo finished" in s.value for s in at.success)
