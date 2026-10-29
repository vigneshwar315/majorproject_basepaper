"""Page A - Overview."""

from __future__ import annotations

import streamlit as st

from frontend import helpers, state, theme
from paper_replication.experiments.reporting import status_counts

WORKFLOW_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.5;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.14,0.08", penwidth=1.2];
  edge [color="#5B6776", arrowsize=0.7];
  req  [label="Requests arrive\\n(pickup + delivery pairs)", fillcolor="#FFFFFF", color="#5B6776"];
  heur [label="Cheapest-insertion heuristic\\nm alternative plans", fillcolor="#E3F3EE", color="#0F6E56"];
  val  [label="Constraint validator\\n(feasibility, deterministic)", fillcolor="#E3F3EE", color="#0F6E56"];
  mat  [label="Evaluation matrix\\nm plans x metrics", fillcolor="#E3F3EE", color="#0F6E56"];
  dmg  [label="DMG: LLM writes a metric\\nfrom a natural-language request", fillcolor="#EEE9FA", color="#5B3FA6"];
  chk  [label="Comprehension verification\\n(static checks, LLM review,\\nexamples with expected values)", fillcolor="#E3F3EE", color="#0F6E56"];
  mcdm [label="MCDM: LLM picks plan\\nfrom trade-off preferences", fillcolor="#EEE9FA", color="#5B3FA6"];
  out  [label="Selected plan + rationale\\n(id and feasibility re-checked)", fillcolor="#FFFFFF", color="#5B6776"];
  req -> heur -> val -> mat -> mcdm -> out;
  dmg -> chk -> mat [label=" new metric", fontsize=9, fontname="Helvetica"];
}
"""


def render() -> None:
    theme.page_header("LLM-enhanced optimization-based decision support",
                      "A reproduction of Ghiani, Manni and Zacchino (IEEE Access 13, 2025): a routing heuristic stays fast and exact about constraints, "
                      "while an LLM turns natural-language requests into new metrics, picks among candidate plans and explains the choice.")
    info = state.provider_info()
    theme.mode_banner(info if "error" not in info else None, info.get("error"))

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("How the system works")
        st.graphviz_chart(WORKFLOW_DOT, width="stretch")
        st.markdown(
            "**The problem.** A fleet starts from one depot. Pickup-and-delivery requests arrive over several days, each with a priority and "
            "sometimes a time window. The company accepts or rejects each request on arrival; accepted requests get a vehicle, a day and a position in a route.\n\n"
            "**What the LLM does, and what it never does.** The LLM writes metric definitions and chooses a plan. Constraints, feasibility, metric values and "
            "every check on the LLM's output are computed by ordinary code. Here, metric definitions use a restricted expression language, so no "
            "generated code is executed."
        )
        st.subheader("How to read this dashboard")
        st.markdown(theme.badge("verified") + theme.badge("llm") + theme.badge("mock") + theme.badge("paper") + theme.badge("extension"), unsafe_allow_html=True)
        st.markdown('<p class="small-note">Each result is tagged with where it comes from. Numbers reported by the paper are never mixed with numbers produced '
                    'by this project unless a table says so explicitly.</p>', unsafe_allow_html=True)
    with right:
        st.subheader("Provider status")
        if "error" in info:
            st.error(info["error"])
        else:
            s = info["settings"]
            c1, c2 = st.columns(2)
            c1.metric("Provider", s["provider"])
            c2.metric("Model", s["model"])
            st.caption("Credentials present: " + ("yes" if s["credentials_present"] else "no") + ". Keys are read from the environment and are never displayed.")
        st.subheader("Latest run")
        runs = helpers.list_runs()
        if not runs:
            st.info("No optimization run yet. Open **Problem configuration**, load an instance and press **Run optimization**.")
        else:
            r = runs[0]
            st.write(f"Run `{r['run_id']}`")
            st.markdown(theme.badge("mock" if r["provider"].get("is_mock") else "llm", "Mock run" if r["provider"].get("is_mock") else "Live LLM run")
                        + (theme.badge("synthetic") if r["instance"].get("synthetic") else ""), unsafe_allow_html=True)
            mc = r.get("mcdm") or {}
            st.caption(f"{r['produced_m']} candidate plans, {r['n_feasible']} feasible. MCDM: {mc.get('status', 'not run')}, selected {mc.get('selected_plan_id') or 'none'}.")
        st.subheader("Experiments and tests")
        sc = status_counts()
        a, b, c = st.columns(3)
        a.metric("Completed", sc["completed"])
        b.metric("Failed", sc["failed"])
        c.metric("Running", sc["running"])
        if sc["total"] == 0:
            st.caption("No experiment has been run yet (they never start automatically).")
        ts = helpers.test_status_summary()
        if ts:
            tt = helpers.test_totals(ts)
            st.caption(f"Last recorded test run ({ts.get('finished_at', '?')}): {tt['passed']} passed, {tt['failed']} failed, {tt['skipped']} skipped. "
                       "Live-provider tests: NOT RUN.")
        else:
            st.caption("No test status recorded yet. Run `python scripts/run_tests.py` to record it.")
        st.subheader("Documents")
        for label in ("Source audit", "Experiment protocol", "Reproduction report"):
            path = helpers.DOCS[label]
            if st.button(f"{label}", key=f"doc_{label}", help=f"Updated {helpers.file_stamp(path)}", width="stretch"):
                st.session_state["doc_to_open"] = label
                pages = st.session_state.get("_pages", {})
                if "logs" in pages:
                    st.switch_page(pages["logs"])
