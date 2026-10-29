"""Page B - Problem configuration and the explicit 'Run optimization' action."""

from __future__ import annotations

import streamlit as st

from frontend import charts, helpers, state, theme
from paper_replication.experiments.prompt_sets import SEED_METRIC_TEXT, TRADEOFF_SEEDS
from paper_replication.pipeline import MetricRequest, run_pipeline
from paper_replication.problem_instance import generate_instance, tiny_instance
from paper_replication.redact import redact
from paper_replication.reference_metrics import PAPER_METRICS
from paper_replication.schemas import InstanceValidationError

ORACLE_NONE = "none (execution-only test, weaker)"


def _load_instance(inst, origin: str) -> None:
    veh, req = helpers.instance_to_frames(inst)
    st.session_state.update(
        instance=inst, instance_origin=origin, vehicles_df=veh, requests_df=req, settings_form=helpers.scalar_settings(inst),
        distance_matrix=inst.distance_matrix, result=None, selected_plan=None, mcdm_alt=None,
        form_version=st.session_state.get("form_version", 0) + 1,
    )


DEMO_PREFERENCE = "Maximize the number of accepted requests. Then keep the total travel distance as short as possible."
DEMO_METRIC = "z5"


def _run_full_demo() -> None:
    """One click: load a synthetic instance, set a preference and one metric request, run the whole pipeline."""
    prov, err = state.current_provider()
    if prov is None:
        st.session_state["last_error"] = f"provider not usable: {err}"
        return
    inst = generate_instance(seed=20251028, n_requests=30, n_vehicles=3)
    _load_instance(inst, "synthetic demonstration (seed 20251028, 30 requests) - one-click demo")
    text = "Introduce a new metric: " + SEED_METRIC_TEXT[DEMO_METRIC][0].lower() + SEED_METRIC_TEXT[DEMO_METRIC][1:]
    st.session_state["preference"] = DEMO_PREFERENCE
    st.session_state["metric_requests"] = [{"text": text, "oracle": DEMO_METRIC}]
    try:
        res = run_pipeline(inst, DEMO_PREFERENCE, prov, m=10, seed=0, max_attempts=5, metric_requests=[MetricRequest(text, DEMO_METRIC)])
        st.session_state.update(result=res, selected_plan=(res.mcdm.selected_plan_id if res.mcdm and res.mcdm.success else None), mcdm_alt=None,
                                last_error=None, demo_done=True)
    except Exception as exc:  # noqa: BLE001 - shown to the user, redacted
        st.session_state["last_error"] = f"{type(exc).__name__}: {redact(exc)}"


def _section_errors(errors_by: dict, section: str) -> None:
    for e in errors_by.get(section, []):
        st.error(str(e), icon=":material/error:")


def render() -> None:
    theme.page_header("Problem configuration", "Define the fleet, the requests and the planning horizon, say what you prefer in plain language, then run the optimization explicitly.")
    info = state.provider_info()
    theme.mode_banner(info if "error" not in info else None, info.get("error"))

    with st.container(border=True):
        d1, d2 = st.columns([3, 1], vertical_alignment="center")
        d1.markdown("**Quick demo.** Loads a 30-request synthetic instance, sets a preference and asks for one new metric (waiting time), "
                    "then runs routing, metric generation and plan selection in one go.")
        live_demo = st.session_state.get("llm_mode") != state.MODE_MOCK
        d2.button("Run full demo", type="primary", width="stretch", on_click=_run_full_demo, disabled=live_demo,
                  help="Available in offline mock mode. With a real provider, use Run optimization below so you can confirm the API calls.")
    if st.session_state.pop("demo_done", False) and st.session_state.get("result") is not None:
        st.success("Demo finished. Look at the map of requests below, then open **Candidate plans**, **Dynamic metric generation** and **Decision explorer**.")
        pages = st.session_state.get("_pages", {})
        if pages:
            l1, l2, l3 = st.columns(3)
            l1.page_link(pages["candidates"], label="Open Candidate plans", icon=":material/alt_route:")
            l2.page_link(pages["dmg"], label="Open Dynamic metric generation", icon=":material/functions:")
            l3.page_link(pages["mcdm"], label="Open Decision explorer", icon=":material/balance:")

    c1, c2, c3 = st.columns([2, 3, 2], gap="medium")
    with c1:
        st.markdown("**Valid example**")
        st.caption("Hand-built instance: 2 vehicles, 3 requests, one time window. Small enough to check by hand.")
        if st.button("Load valid example instance", width="stretch"):
            _load_instance(tiny_instance(), "valid example (hand-built)")
            st.rerun()
    with c2:
        st.markdown("**Synthetic demonstration instance**")
        a, b, c = st.columns(3)
        seed = a.number_input("Seed", 0, 10**9, 20251028, key="gen_seed")
        n_req = b.number_input("Requests", 0, 200, 60, key="gen_n")
        n_veh = c.number_input("Vehicles", 1, 10, 3, key="gen_v")
        if st.button("Load synthetic demonstration instance", width="stretch"):
            _load_instance(generate_instance(seed=int(seed), n_requests=int(n_req), n_vehicles=int(n_veh)), f"synthetic demonstration (seed {int(seed)})")
            st.rerun()
    with c3:
        st.markdown("**Current instance**")
        inst = st.session_state.get("instance")
        if inst is None:
            st.caption("None loaded yet.")
        else:
            st.markdown(theme.badge("synthetic") if inst.synthetic else theme.badge("verified", "Hand-built example"), unsafe_allow_html=True)
            st.caption(st.session_state.get("instance_origin", ""))

    if st.session_state.get("instance") is None:
        st.info("Load the valid example or a synthetic demonstration instance to start. Nothing runs until you press **Run optimization**.")
        return

    ver = st.session_state.get("form_version", 0)
    tabs = st.tabs(["Map of requests", "Fleet and depot", "Requests", "Horizon and travel cost", "Preferences and metrics"])
    map_tab, tabs = tabs[0], tabs[1:]
    sf = dict(st.session_state["settings_form"])

    with tabs[0]:
        d1, d2, d3 = st.columns(3)
        sf["depot_x"] = d1.number_input("Depot x", value=float(sf["depot_x"]), key=f"dx{ver}")
        sf["depot_y"] = d2.number_input("Depot y", value=float(sf["depot_y"]), key=f"dy{ver}")
        sf["depot_zip"] = d3.text_input("Depot ZIP", value=str(sf["depot_zip"]), key=f"dz{ver}")
        st.markdown("**Vehicles** (add or remove rows; capacity is the maximum load at any moment)")
        veh_edit = st.data_editor(
            st.session_state["vehicles_df"], num_rows="dynamic", hide_index=True, width="stretch", key=f"veh_editor_{ver}",
            column_config={"id": st.column_config.TextColumn("Vehicle id", required=True),
                           "capacity": st.column_config.NumberColumn("Capacity", min_value=1, step=1, required=True),
                           "shift_duration_min": st.column_config.NumberColumn("Max route duration (min)", min_value=1, help="Driver shift limit (the paper mentions 8 h = 480 min).", required=True)})

    with tabs[1]:
        st.caption("Times use HH:MM. Leave a time window empty if the request has none. Priority is 'normal' or 'high'. Pickup always precedes delivery on the same route.")
        req_edit = st.data_editor(
            st.session_state["requests_df"], num_rows="dynamic", hide_index=True, width="stretch", key=f"req_editor_{ver}", height=360,
            column_config={
                "id": st.column_config.TextColumn("Request", required=True), "customer": st.column_config.TextColumn("Customer"),
                "arrival_day": st.column_config.NumberColumn("Arrival day", min_value=0, step=1, required=True),
                "arrival_seq": st.column_config.NumberColumn("Arrival order", min_value=0, step=1),
                "pickup_x": st.column_config.NumberColumn("Pickup x"), "pickup_y": st.column_config.NumberColumn("Pickup y"), "pickup_zip": st.column_config.TextColumn("Pickup ZIP"),
                "delivery_x": st.column_config.NumberColumn("Delivery x"), "delivery_y": st.column_config.NumberColumn("Delivery y"), "delivery_zip": st.column_config.TextColumn("Delivery ZIP"),
                "demand": st.column_config.NumberColumn("Demand", min_value=1, step=1, required=True),
                "priority": st.column_config.SelectboxColumn("Priority", options=["normal", "high"], required=True),
                "pickup_tw_from": st.column_config.TextColumn("Pickup window from"), "pickup_tw_to": st.column_config.TextColumn("to"),
                "delivery_tw_from": st.column_config.TextColumn("Delivery window from"), "delivery_tw_to": st.column_config.TextColumn("to"),
                "service_time_min": st.column_config.NumberColumn("Service (min)", min_value=0.0, help="The paper's timing formula implies zero service time."),
            })

    with tabs[2]:
        h1, h2, h3 = st.columns(3)
        sf["horizon_days"] = h1.number_input("Planning horizon (days)", 1, 31, int(sf["horizon_days"]), key=f"hd{ver}")
        sf["current_day"] = h2.number_input("Current day ('today')", 0, 30, int(sf["current_day"]), key=f"cd{ver}", help="Requests arrive on days 0 to this day; 'today' metrics refer to it.")
        sf["max_lead_days"] = h3.number_input("Max days ahead a request may be scheduled", 0, 10, int(sf["max_lead_days"]), key=f"ml{ver}")
        t1, t2, t3 = st.columns(3)
        sf["shift_start"] = t1.text_input("Depot opens (HH:MM)", str(sf["shift_start"]), key=f"ss{ver}")
        sf["depot_close"] = t2.text_input("Depot closes (HH:MM)", str(sf["depot_close"]), key=f"dc{ver}")
        sf["minutes_per_unit"] = t3.number_input("Minutes of driving per distance unit", 0.01, 100.0, float(sf["minutes_per_unit"]), key=f"mp{ver}")
        st.markdown("**Travel cost**")
        mode = st.radio("Distance model", ["Euclidean distance between coordinates", "Distance matrix from CSV"], horizontal=True, key=f"dm_mode{ver}",
                        help="With a CSV, every pair of nodes needs a non-negative distance. Node ids: depot, <request>:P (pickup), <request>:D (delivery).")
        if mode.startswith("Distance"):
            up = st.file_uploader("Upload distance matrix (square CSV, header and first column = node ids)", type=["csv"], key=f"dm_up{ver}")
            if up is not None:
                mat, perrs = helpers.parse_distance_matrix_csv(up.getvalue().decode("utf-8", errors="replace"))
                st.session_state["distance_matrix"] = mat
                for e in perrs[:5]:
                    st.error(e)
            if st.session_state.get("instance") is not None:
                st.download_button("Download a template matrix for this instance", helpers.distance_matrix_template_csv(st.session_state["instance"]), "distance_matrix_template.csv", "text/csv")
            matrix = st.session_state.get("distance_matrix")
        else:
            matrix = None
            st.session_state["distance_matrix"] = None

    # ---- validation (no backend call until the user presses Run) --------------------------------------------
    new_inst, errors = helpers.frames_to_instance(sf, veh_edit, req_edit, matrix)
    by = helpers.errors_by_section(errors)
    with tabs[0]:
        _section_errors(by, "depot")
        _section_errors(by, "vehicles")
    with tabs[1]:
        _section_errors(by, "requests")
    with tabs[2]:
        _section_errors(by, "horizon")
        _section_errors(by, "travel")

    with map_tab:
        shown = new_inst or st.session_state.get("instance")
        st.plotly_chart(charts.instance_figure(shown), width="stretch")
        nh = sum(1 for r in shown.requests if r.priority == 1)
        ntw = sum(1 for r in shown.requests if r.pickup_tw or r.delivery_tw)
        st.caption(f"{len(shown.requests)} requests ({nh} high priority, {ntw} with a time window), {len(shown.vehicles)} vehicles, depot at "
                   f"({shown.depot.x:g}, {shown.depot.y:g}). Hover a point for details. Coordinates are abstract (synthetic), not a real map. "
                   "Edit points in the **Requests** tab.")

    with tabs[3]:
        st.markdown("**Trade-off preference** (natural language; used by the MCDM step)")
        seed_choice = st.selectbox("Start from a paper trade-off prompt (Table 4)", ["(write your own)"] + [f"Prompt {k}" for k in TRADEOFF_SEEDS], key="seed_pref_choice")
        if seed_choice != "(write your own)" and st.session_state.get("_last_seed_choice") != seed_choice:
            st.session_state["preference"] = TRADEOFF_SEEDS[int(seed_choice.split()[1])]
            st.session_state["_last_seed_choice"] = seed_choice
        st.text_area("Preference", key="preference", height=110, placeholder="e.g. Maximize the number of accepted requests, but keep the routes balanced across vehicles.")
        st.markdown("**New metrics** (optional; each is turned into a validated metric by the DMG module)")
        st.caption("A metric request needs expected results to be tested properly. Pick a reference metric as the oracle, or run without one (weaker test, clearly flagged).")
        reqs = st.session_state["metric_requests"]
        pick = st.selectbox("Add a metric request from the paper (Table 2)", ["(choose)"] + [f"{m}: {SEED_METRIC_TEXT[m][:70]}" for m in PAPER_METRICS], key="metric_seed_pick")
        cA, cB = st.columns([1, 1])
        if cA.button("Add this metric request") and pick != "(choose)":
            mid = pick.split(":")[0]
            reqs.append({"text": "Introduce a new metric: " + SEED_METRIC_TEXT[mid][0].lower() + SEED_METRIC_TEXT[mid][1:], "oracle": mid})
            st.rerun()
        if cB.button("Add an empty metric request"):
            reqs.append({"text": "", "oracle": ORACLE_NONE})
            st.rerun()
        for i, rq in enumerate(list(reqs)):
            with st.container(border=True):
                rq["text"] = st.text_area(f"Metric request d{i + 1}", rq["text"], key=f"mr_text_{i}", height=80)
                opts = [ORACLE_NONE] + PAPER_METRICS
                cur = rq["oracle"] if rq["oracle"] in opts else ORACLE_NONE
                rq["oracle"] = st.selectbox("Expected results come from", opts, index=opts.index(cur), key=f"mr_or_{i}")
                if st.button("Remove", key=f"mr_rm_{i}"):
                    reqs.pop(i)
                    st.rerun()

    st.divider()
    st.subheader("Run")
    r1, r2, r3 = st.columns(3)
    m = r1.number_input("Candidate plans (m)", 2, 100, 10, key="run_m", help="The paper evaluates m = 10, 50 and 100 in its MCDM tests.")
    rseed = r2.number_input("Random seed", 0, 10**9, 0, key="run_seed")
    max_att = r3.number_input("Max DMG attempts", 1, 10, 5, key="run_att")
    prov, perr = state.current_provider()
    pref_ok = bool(st.session_state.get("preference", "").strip())
    live = prov is not None and not prov.is_mock
    n_calls = 1 + sum(2 * int(max_att) for _ in reqs)
    ack = True
    if live:
        st.warning(f"This run calls {prov.name} up to about {n_calls} time(s) with model {prov.model}. API usage may be billed.")
        ack = st.checkbox("I understand and want to call the provider", key="live_ack")
    problems = []
    if errors:
        problems.append(f"{len(errors)} input problem(s) above")
    if prov is None:
        problems.append("provider not usable: " + (perr or ""))
    if not pref_ok:
        problems.append("preference text is empty")
    if any(not rq["text"].strip() for rq in reqs):
        problems.append("a metric request is empty")
    if problems:
        st.info("Run is disabled: " + "; ".join(problems) + ".")
    if st.button("Run optimization", type="primary", disabled=bool(problems) or not ack):
        assert new_inst is not None and prov is not None
        try:
            with st.status("Running the pipeline ...", expanded=True) as status:
                res = run_pipeline(
                    new_inst, st.session_state["preference"], prov, m=int(m), seed=int(rseed), max_attempts=int(max_att),
                    metric_requests=[MetricRequest(rq["text"], None if rq["oracle"] == ORACLE_NONE else rq["oracle"]) for rq in reqs],
                    progress=lambda msg: st.write(msg),
                )
                status.update(label="Run finished", state="complete")
            st.session_state.update(instance=new_inst, result=res, selected_plan=(res.mcdm.selected_plan_id if res.mcdm and res.mcdm.success else None), mcdm_alt=None, last_error=None)
        except InstanceValidationError as exc:
            st.session_state["last_error"] = "; ".join(exc.errors)
        except Exception as exc:  # noqa: BLE001 - shown to the user, redacted
            st.session_state["last_error"] = f"{type(exc).__name__}: {redact(exc)}"
    if st.session_state.get("last_error"):
        st.error("The run did not complete. " + st.session_state["last_error"])
    res = st.session_state.get("result")
    if res is not None:
        st.success(f"Run {res.run_id} completed: {res.candidate_set.produced_m} candidate plans, "
                   f"MCDM {res.mcdm.status if res.mcdm else 'not run'}. Results are in **Candidate plans**, **Dynamic metric generation** and **Decision explorer**.")
        for w in res.warnings:
            st.warning(w)
