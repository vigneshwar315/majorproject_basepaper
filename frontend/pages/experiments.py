"""Page F - Experiments and analytics (C1-C6). Status and numbers are read from files on disk."""

from __future__ import annotations

import json

import streamlit as st

from frontend import charts, helpers, state, theme
from paper_replication.experiments.reporting import (
    comparison_with_paper,
    list_experiment_runs,
    load_experiment_run,
    status_counts,
)
from paper_replication.experiments.runner import EXPERIMENTS, run_experiment
from paper_replication.redact import redact


def _summary_charts(exp_id: str, s: dict) -> None:
    if exp_id == "C1" and s.get("per_type"):
        types = list(s["per_type"])
        st.plotly_chart(charts.grouped_bars(types, {"generated (of 30)": [s["per_type"][t].get("generated") for t in types], "tests passed (of 30)": [s["per_type"][t].get("tests_passed") for t in types]},
                                            "Metrics generated and passing the example tests", "prompts (of 30)"), width="stretch")
    elif exp_id == "C2" and s.get("per_m"):
        ms = list(s["per_m"])
        st.plotly_chart(charts.bar_pair([f"m={m}" for m in ms], [s["per_m"][m].get("correct_per_30_mean") for m in ms], "Correct selections per 30 prompts", "correct (of 30)"), width="stretch")
    elif exp_id == "C4" and s.get("per_level"):
        lv = list(s["per_level"])
        st.plotly_chart(charts.bar_pair([f"level {x}" for x in lv], [s["per_level"][x].get("correct_per_30_mean") for x in lv], "Correct selections by prompt degradation", "correct (of 30)"), width="stretch")
    elif exp_id == "C3" and s.get("cells"):
        cells = s["cells"]
        types = list(cells)
        levels = sorted({lv for t in types for lv in cells[t]})
        st.plotly_chart(charts.grouped_bars(types, {lv: [cells[t].get(lv, {}).get("passed") for t in types] for lv in levels}, "Metrics passing the tests by prompt quality", "passed (of 30)"), width="stretch")
    elif exp_id == "C5" and s.get("mcdm"):
        ms = list(s["mcdm"])
        st.plotly_chart(charts.bar_pair([f"m={m}" for m in ms], [s["mcdm"][m].get("mean_time_s") for m in ms], "Decision step wall-clock time", "seconds"), width="stretch")


def render() -> None:
    theme.page_header("Experiments", "The six tests of Section IV-B, rebuilt for this project. Numbers on this page are read from result files; nothing is typed in by hand.")
    theme.provenance("mock", "paper", "verified")
    counts = status_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs on disk", counts["total"])
    c2.metric("Completed", counts["completed"])
    c3.metric("Failed", counts["failed"])
    c4.metric("Running", counts["running"])
    prov, err = state.current_provider()
    theme.mode_banner(state.provider_info(), err)

    st.subheader("Run an experiment")
    exp_id = st.selectbox("Experiment", list(EXPERIMENTS), format_func=lambda k: f"{k}: {EXPERIMENTS[k].title} ({EXPERIMENTS[k].source})")
    spec = EXPERIMENTS[exp_id]
    st.write(spec.question)
    st.caption(spec.design)
    with st.expander("Caveats that apply to this experiment"):
        for c in spec.caveats:
            st.write(f"- {c}")
    live = prov is not None and not prov.is_mock
    est = helpers.prompt_call_estimate(exp_id)
    if live:
        st.warning(f"About {est} calls to {prov.name} / {prov.model}. This can take time and cost money.")
        confirmed = st.checkbox("I confirm I want to run this against the live provider", key="exp_confirm")
    else:
        confirmed = True
        st.caption("With the mock provider the run is a mechanics check (category: synthetic_demonstration). Larger experiments can take up to a minute.")
    if st.button("Run experiment", type="primary", disabled=prov is None or not confirmed):
        bar = st.progress(0.0, text="Starting")

        def prog(i: int, n: int, msg: str) -> None:
            bar.progress(min(1.0, i / max(n, 1)), text=redact(msg))

        try:
            r = run_experiment(exp_id, prov, progress=prog)
            st.session_state["last_exp"] = (r.exp_id, r.run_id)
            bar.progress(1.0, text="Finished")
            st.success(f"Run {r.run_id} saved. Category: {r.category}")
        except Exception as exc:  # noqa: BLE001
            st.error(redact(f"{type(exc).__name__}: {exc}"))

    st.subheader("Runs on disk")
    runs = list_experiment_runs()
    if runs.empty:
        st.info("No experiment has been run yet. Choose one above and press **Run experiment**.")
        return
    st.dataframe(runs, hide_index=True, width="stretch")
    last = st.session_state.get("last_exp")
    labels = [f"{r.exp_id} / {r.run_id}" for r in runs.itertuples()]
    idx = labels.index(f"{last[0]} / {last[1]}") if last and f"{last[0]} / {last[1]}" in labels else 0
    pick = st.selectbox("Inspect a run", labels, index=idx)
    e_id, r_id = pick.split(" / ")
    data = load_experiment_run(e_id, r_id)
    meta, summ, rows = data["meta"] or {}, data["summary"], data["results"]
    category = meta.get("category", "")
    if category == "synthetic_demonstration":
        st.markdown(theme.badge("mock", "Synthetic demonstration") + " Mock provider and/or synthetic data: these numbers describe the software, not language-model quality.", unsafe_allow_html=True)
    elif category:
        st.markdown(theme.badge("llm", category), unsafe_allow_html=True)
    if data["status"] and data["status"].get("status") == "failed":
        st.error(redact(str(data["status"].get("error", "failed"))))
    if summ:
        _summary_charts(e_id, summ)
        with st.expander("Summary (JSON)"):
            st.json(summ, expanded=False)
        st.subheader("Comparison with the paper")
        table, valid, reason = comparison_with_paper(e_id, summ, category)
        (st.info if not valid else st.warning)(reason)
        if valid and not table.empty:
            st.markdown(theme.badge("paper", "Paper value") + theme.badge("llm", "This run"), unsafe_allow_html=True)
            st.dataframe(table, hide_index=True, width="stretch")
        elif not table.empty:
            with st.expander("Paper's reported values alone (reference)"):
                st.dataframe(table[[c for c in table.columns if c != "this run"]], hide_index=True, width="stretch")
    if rows is not None:
        st.subheader("Per-item results")
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button("Download results (CSV)", rows.to_csv(index=False), f"{e_id}_{r_id}.csv", "text/csv")
    if summ:
        st.download_button("Download summary (JSON)", json.dumps(summ, indent=2), f"{e_id}_{r_id}_summary.json", "application/json")
