"""Page D - Dynamic metric generation: spec, attempts, evaluated values, DSL validator and a tester."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend import state, theme
from paper_replication.dmg.evaluation import EvalError, evaluate_spec
from paper_replication.dmg.examples import expected_for, get_examples
from paper_replication.dmg.generation import DMGPrompt, run_dmg
from paper_replication.dmg.metric_specification import SpecError, dsl_reference
from paper_replication.dmg.reference_specs import parse_spec, reference_spec
from paper_replication.reference_metrics import ALL_METRICS, DESCRIPTIONS


def _trace_view(r) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Outcome", r.status)
    c2.metric("Attempts used", f"{r.attempts_used} / {r.max_attempts}")
    c3.metric("Provider", f"{r.provider}{' (mock)' if r.is_mock else ''}")
    if r.warning_to_supervisor:
        st.warning(r.warning_to_supervisor)
    if r.spec:
        st.markdown(theme.badge("llm", "Specification written by the language model") + theme.badge("verified", "Passed static check, comprehension check and example tests"), unsafe_allow_html=True)
        st.code(r.spec.to_json(indent=2), language="json")
    rows = [{"attempt": t.attempt, "outcome": t.outcome, "static errors": "; ".join(t.static_errors), "comprehension check": t.llm_verdict or "", "execution error": t.execution_error or "",
             "values on the 3 examples": ", ".join(f"{v:.6g}" for v in t.execution_values) if t.execution_values else "", "error fed back": t.error or ""} for t in r.trace]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for t in r.trace:
        with st.expander(f"Attempt {t.attempt}: {t.outcome}"):
            st.caption("Specification produced in this attempt (model output, untrusted until validated):")
            st.code(t.spec_json or "(none)", language="json")
            if t.llm_reason:
                theme.llm_box(t.llm_reason, r.is_mock)


def render() -> None:
    theme.page_header("Dynamic metric generation", "How a new metric request becomes a checked, executable metric. The language model writes a specification; code checks it and runs it.")
    theme.provenance("llm", "verified")
    st.graphviz_chart('digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#EEF2F6", color="#D5DCE4", fontname="Helvetica"]; '
                      'a [label="Metric request\\n(text)"]; b [label="LLM writes\\nspecification", fillcolor="#E6E0F5"]; c [label="Static check\\n(code)", fillcolor="#DDF1EC"]; '
                      'd [label="LLM comprehension\\ncheck", fillcolor="#E6E0F5"]; e [label="Run on 3 examples\\nvs expected (code)", fillcolor="#DDF1EC"]; f [label="Metric accepted", fillcolor="#DDF1EC"]; '
                      'g [label="Retry with error\\n(max attempts)", fillcolor="#FBF0D9"]; a->b->c->d->e->f; c->g; d->g; e->g; g->b; }')
    st.caption("Algorithm 1 of the paper. Deviation U01: the model returns a JSON expression tree, not Python code, so no model output is ever executed as code.")

    res = st.session_state.get("result")
    t1, t2, t3, t4 = st.tabs(["Results of the last run", "Try one metric request", "Check a specification", "Language reference"])
    with t1:
        if res is None or not res.dmg_results:
            st.info("The last run has no generated metrics. On **Problem configuration**, press **Run full demo** (it adds one metric request), "
                    "or add a metric request in the **Preferences and metrics** tab before **Run optimization**. You can also use the **Try one metric request** tab here.")
        else:
            for key, r in res.dmg_results.items():
                with st.expander(f"{key}: {r.status}" + (f" ({r.spec.name})" if r.spec else ""), expanded=len(res.dmg_results) == 1):
                    _trace_view(r)
            if res.matrix:
                cols = [m for m in res.metric_ids if m in res.dynamic_metric_texts]
                if cols:
                    st.subheader("Values of generated metrics on the candidate plans")
                    df = pd.DataFrame([{"plan": pid, **{c: v.get(c) for c in cols}} for pid, v in res.matrix.items()])
                    st.dataframe(df, hide_index=True, width="stretch")
    with t2:
        st.write("Ask for a metric and watch Algorithm 1 run. The expected values come from the reference implementation of a Table 2 metric, so the test has a real oracle.")
        mid = st.selectbox("Reference metric used as the oracle", ALL_METRICS, format_func=lambda m: f"{m}: {DESCRIPTIONS[m][:70]}")
        text = st.text_area("Metric request", value=DESCRIPTIONS[mid], height=100)
        attempts = st.number_input("Maximum attempts", 1, 8, 5)
        prov, err = state.current_provider()
        theme.mode_banner(state.provider_info(), err)
        live = prov is not None and not prov.is_mock
        ok = st.checkbox("I understand this sends my request to the configured provider", value=not live, disabled=not live) if live else True
        if st.button("Generate metric", type="primary", disabled=prov is None or not ok):
            with st.spinner("Running Algorithm 1"):
                try:
                    exs = get_examples()
                    r = run_dmg(DMGPrompt(text, [e.view for e in exs], expected_for(mid, exs)), prov, int(attempts))
                    st.session_state["dmg_tester"] = r
                except Exception as exc:  # noqa: BLE001 - shown to the user, redacted
                    from paper_replication.redact import redact

                    st.session_state["dmg_tester"] = None
                    st.error(redact(f"{type(exc).__name__}: {exc}"))
        if st.session_state.get("dmg_tester") is not None:
            _trace_view(st.session_state["dmg_tester"])
    with t3:
        st.write("Paste a metric specification (JSON). The checker applies the same static rules as the pipeline, then evaluates it on the three hand-built example plans.")
        default = reference_spec("z1", "Accepted requests divided by all received requests").to_json(indent=2)
        src = st.text_area("Specification", value=default, height=260, key="spec_src")
        if st.button("Check specification"):
            try:
                spec = parse_spec(src)
                st.success("The specification is valid.")
                vals = []
                for e in get_examples():
                    try:
                        vals.append({"example": e.name, "value": evaluate_spec(spec, e.view), "note": e.note})
                    except EvalError as exc:
                        vals.append({"example": e.name, "value": None, "note": f"evaluation error: {exc}"})
                st.dataframe(pd.DataFrame(vals), hide_index=True, width="stretch")
            except SpecError as exc:
                st.error("The specification was rejected:")
                for m in exc.errors:
                    st.write(f"- {m}")
    with t4:
        st.code(dsl_reference(), language="text")
