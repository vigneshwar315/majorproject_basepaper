"""Page E - Decision explorer: multi-criteria selection with checks on the model's answer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import charts, state, theme
from paper_replication.experiments.paper_reference import table9_matrix
from paper_replication.experiments.prompt_sets import TRADEOFF_SEEDS
from paper_replication.mcdm.reference_rules import RULES, acceptable_plans
from paper_replication.mcdm.selection import Candidate, run_mcdm


def _show(r, matrix: dict, ids: list[str], source_label: str) -> None:
    if r.success:
        st.success(f"Selected plan: {r.selected_plan_id}")
    else:
        st.error(f"The decision step failed: {r.error}")
    if r.warning_to_supervisor:
        st.warning(r.warning_to_supervisor)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(theme.badge("verified", "Checked by code") , unsafe_allow_html=True)
        v = r.validation
        if v:
            st.write(f"Output checks passed: **{'yes' if v.ok else 'no'}**")
            for e in v.errors:
                st.error(e)
            if v.claims_checked:
                st.write(f"Numeric claims in the explanation: {v.claims_supported} of {v.claims_checked} match the matrix.")
            for u in v.unsupported_claims:
                st.warning(f"Unsupported claim: {u}")
        st.write(f"Candidates: {r.n_candidates} ({r.n_infeasible_excluded} infeasible excluded before the model saw the matrix)")
        st.write("Criteria named by the model: " + (", ".join(r.criteria_used) or "none"))
        if r.selected_plan_id in matrix:
            st.dataframe(pd.DataFrame([{"metric": m, "value": matrix[r.selected_plan_id].get(m)} for m in ids]), hide_index=True, width="stretch")
    with c2:
        theme.llm_box(r.justification or "(no explanation)", r.is_mock)
    st.caption(source_label)


def render() -> None:
    theme.page_header("Decision explorer", "Pick one plan from the candidates using a preference written in plain language. Code excludes infeasible plans and checks the answer.")
    theme.provenance("llm", "verified")
    res = st.session_state.get("result")
    tab1, tab2 = st.tabs(["Plans of the last run", "Paper's example matrix (Table 9)"])
    prov, err = state.current_provider()
    with tab1:
        if res is None or not res.matrix:
            st.info("Run an optimization first on **Problem configuration**.")
        else:
            pref = st.text_area("Preference", value=st.session_state.get("mcdm_pref") or res.preference, height=90, key="mcdm_pref_box")
            st.caption("Examples: " + " | ".join(f"“{t}”" for t in list(TRADEOFF_SEEDS.values())[:2]) if isinstance(TRADEOFF_SEEDS, dict) else "")
            theme.mode_banner(state.provider_info(), err)
            if st.button("Choose a plan", type="primary", disabled=prov is None or not pref.strip()):
                cands = [Candidate(p.plan_id, res.matrix[p.plan_id], True) for p in res.plans if p.plan_id in res.matrix]
                cands += [Candidate(p.plan_id, {}, False) for p in res.plans if p.plan_id not in res.matrix]
                try:
                    st.session_state["mcdm_alt"] = run_mcdm(cands, pref, prov, res.metric_ids, extra_descriptions=res.dynamic_metric_texts)
                except Exception as exc:  # noqa: BLE001
                    from paper_replication.redact import redact

                    st.error(redact(f"{type(exc).__name__}: {exc}"))
            alt = st.session_state.get("mcdm_alt")
            shown = alt if alt is not None else res.mcdm
            if shown is not None and shown.n_candidates == len(res.plans):
                _show(shown, res.matrix, res.metric_ids, "The explanation is model text. Only the checks on the left are computed.")
                if shown.success and alt is not None and st.button("Use this plan as the selected plan"):
                    st.session_state["selected_plan"] = shown.selected_plan_id
                    st.rerun()
            df = pd.DataFrame([{"plan": pid, **v} for pid, v in res.matrix.items()])
            st.plotly_chart(charts.parallel_figure(df.assign(feasible=True), [m for m in res.metric_ids][:8], shown.selected_plan_id if shown else None), width="stretch")
    with tab2:
        ids, matrix = table9_matrix()
        st.markdown(theme.badge("paper", "Evaluation matrix transcribed from the paper (Table 9)"), unsafe_allow_html=True)
        st.caption("This matrix is an input provided by the paper, not something this project produced.")
        st.dataframe(pd.DataFrame([{"plan": p, **v} for p, v in matrix.items()]), hide_index=True, width="stretch")
        opts = {f"Rule {k}: " + str(TRADEOFF_SEEDS.get(k, "")) if isinstance(TRADEOFF_SEEDS, dict) else f"Rule {k}": k for k in RULES}
        pick = st.selectbox("Trade-off prompt", list(opts))
        ptext = st.text_area("Preference text", value=str(TRADEOFF_SEEDS.get(opts[pick], "")) if isinstance(TRADEOFF_SEEDS, dict) else "", key="t9_pref")
        rule = opts[pick]
        if all(stage.metric in ids for stage in RULES[rule]):
            acc = acceptable_plans(rule, matrix)
            st.write("Plans accepted by our reference rule (a reconstruction, see deviation U15): " + (", ".join(acc) or "none"))
        else:
            acc = []
            st.info("Our reference rule for this prompt uses extension metrics (z11 to z14) that Table 9 does not contain, so no reference answer is shown.")
        if st.button("Choose from the Table 9 matrix", disabled=prov is None or not ptext.strip()):
            try:
                r = run_mcdm([Candidate(p, v, True) for p, v in matrix.items()], ptext, prov, ids)
                _show(r, matrix, ids, "Selection on the paper's matrix with this project's prompts.")
                if r.success and acc:
                    st.write("Agrees with the reference rule: " + ("yes" if r.selected_plan_id in acc else "no"))
            except Exception as exc:  # noqa: BLE001
                from paper_replication.redact import redact

                st.error(redact(f"{type(exc).__name__}: {exc}"))
