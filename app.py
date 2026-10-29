"""Streamlit entry point:  streamlit run app.py

UI only. All computation is done by the ``paper_replication`` package (src/) through ``run_pipeline`` and the
experiment runner; nothing expensive runs when a page loads.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402
from frontend import state, theme  # noqa: E402
from frontend.pages import candidates, dmg, experiments, logs_reports, mcdm, overview, problem  # noqa: E402

st.set_page_config(page_title="LLM-enhanced DSS reproduction", page_icon=":material/route:", layout="wide", initial_sidebar_state="expanded")
theme.inject_css()
state.init_state()

PAGES = [
    st.Page(overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(problem.render, title="Problem configuration", icon=":material/tune:", url_path="problem"),
    st.Page(candidates.render, title="Candidate plans", icon=":material/alt_route:", url_path="candidates"),
    st.Page(dmg.render, title="Dynamic metric generation", icon=":material/functions:", url_path="dmg"),
    st.Page(mcdm.render, title="Decision explorer", icon=":material/balance:", url_path="mcdm"),
    st.Page(experiments.render, title="Experiments and analytics", icon=":material/science:", url_path="experiments"),
    st.Page(logs_reports.render, title="Logs and reports", icon=":material/description:", url_path="logs"),
]

st.session_state["_pages"] = {"overview": PAGES[0], "problem": PAGES[1], "candidates": PAGES[2], "dmg": PAGES[3], "mcdm": PAGES[4], "experiments": PAGES[5], "logs": PAGES[6]}

with st.sidebar:
    st.markdown("### Reproduction study")
    st.caption("Ghiani, Manni & Zacchino, IEEE Access 13 (2025). Optimization-based decision support with LLMs.")
    st.radio("LLM mode", [state.MODE_MOCK, state.MODE_CONFIGURED], key="llm_mode",
             help="Mock mode needs no credentials and never calls a network service. 'Provider from .env' uses LLM_PROVIDER / LLM_MODEL and your local API key.")
    prov, err = state.current_provider()
    if prov is not None:
        st.caption(("Mock provider, not an LLM" if prov.is_mock else f"{prov.name} / {prov.model}") + (" (offline)" if prov.is_mock else ""))
    else:
        st.error(err or "provider unavailable")

st.navigation(PAGES).run()
