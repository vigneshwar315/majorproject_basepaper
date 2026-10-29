"""Page G - Logs, documents and reports (all text passes through the redactor)."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from frontend import helpers, theme
from paper_replication.config import LOGS_DIR
from paper_replication.redact import redact
from paper_replication.runlog import read_log


def render() -> None:
    theme.page_header("Logs and reports", "Project documents, the run log and the real test status. Secrets are removed from everything shown here.")
    t1, t2, t3, t4 = st.tabs(["Documents", "Run log", "Saved runs", "Test status"])
    with t1:
        names = list(helpers.DOCS)
        want = st.session_state.get("doc_to_open")
        name = st.selectbox("Document", names, index=names.index(want) if want in names else 0)
        path = helpers.DOCS[name]
        st.caption(f"{path.name}, last modified {helpers.file_stamp(path)}")
        text = helpers.read_doc(path)
        if text is None:
            st.info("This document has not been created yet.")
        elif path.suffix == ".csv":
            st.dataframe(pd.read_csv(io.StringIO(text)), hide_index=True, width="stretch")
        else:
            st.markdown(text)
        if text:
            st.download_button("Download", text, path.name)
    with t2:
        rows = read_log()
        if not rows:
            st.info("The run log is empty. It fills when you run an optimization or an experiment.")
        else:
            df = pd.DataFrame(rows)
            kinds = sorted(df["event"].dropna().unique()) if "event" in df else []
            sel = st.multiselect("Event types", kinds, default=kinds)
            if "event" in df and sel:
                df = df[df["event"].isin(sel)]
            st.dataframe(df.astype(str).map(redact), hide_index=True, width="stretch")
            st.caption(f"{len(df)} entries from logs/run_log.jsonl")
    with t3:
        runs = helpers.list_runs()
        if not runs:
            st.info("No optimization run has been saved yet.")
        else:
            st.dataframe(pd.DataFrame([{"run": r.get("run_id"), "created": r.get("created_at"), "category": r.get("category"), "provider": (r.get("provider") or {}).get("provider"),
                                        "plans": r.get("produced_m"), "feasible": r.get("n_feasible"), "selected": (r.get("mcdm") or {}).get("selected_plan_id")} for r in runs]),
                         hide_index=True, width="stretch")
    with t4:
        ts = helpers.test_status_summary()
        if ts is None:
            st.info("No test status file yet. Run `python scripts/run_tests.py` from the project folder; it writes logs/test_status.json.")
        else:
            st.write(f"Last run: {ts.get('finished_at', 'unknown')}")
            st.dataframe(pd.DataFrame([{"suite": k, **v} for k, v in (ts.get("suites") or {}).items()]), hide_index=True, width="stretch")
            for n in ts.get("notes", []):
                st.caption(n)
        log = LOGS_DIR / "repair_log.md"
        if log.exists():
            with st.expander("Repair log"):
                st.markdown(redact(log.read_text(encoding="utf-8")))
