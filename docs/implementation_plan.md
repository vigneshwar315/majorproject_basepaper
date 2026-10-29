# Implementation Plan

## 1. Architecture

```
frontend/ + app.py (Streamlit/Plotly)   ──calls──▶  paper_replication.pipeline / experiments.runner
                                                      │
 problem_instance ─▶ insertion_heuristic ─▶ candidate_generation ─▶ constraints (validator)
                                                      │
                        dmg (spec → static validation → LLM check → execute on examples)
                                                      │
                        mcdm (prompt → provider → validation → selection + justification)
                                                      │
                        llm/ (base interface, gemini, openai, mock)   config / redact / runlog
```
The research package `src/paper_replication` has **no** Streamlit import; the frontend only calls its public functions.

## 2. Data schemas (`schemas.py`, dataclasses with explicit validation)
`Location(id,x,y,zip)`, `Vehicle(id,capacity,shift_duration)`, `Request(id,arrival_day,arrival_seq,pickup,delivery,demand,priority,pickup_tw,delivery_tw,service_time)`, `Instance(depot,vehicles,requests,horizon_days,current_day,shift_start,depot_close,minutes_per_unit,max_lead_days)`, `PolicyParams`, `Stop(request_id,kind)`, `Route(day,vehicle_id,stops)`, `Plan(plan_id,policy,routes,decisions,feasible,violations)`, `RouteTiming`, metric spec / MCDM result / run record types.

## 3. Algorithm-to-paper mapping
See `requirements_traceability.csv`. Key mapping: Algorithm 1 → `dmg/generation.py::run_dmg`; Sec. III-B/D → `mcdm/selection.py::run_mcdm`; IV-B1 timing formulae → `constraints.py::compute_route_timing`.

## 4. Constraint validation (`constraints.py`)
Independent of the heuristic. `validate_plan(instance, plan)` re-derives timing and checks: unique/known ids, pickup before delivery on the same route and day, capacity prefix-load ≤ Q, time windows (service start ≤ l), route duration ≤ shift duration, return ≤ depot close, day ∈ [arrival_day, arrival_day+max_lead_days] ∩ [0,horizon), every accepted request appears exactly once (pickup and delivery), rejected requests appear nowhere. A plan's `feasible` flag is only ever set by this function.

## 5. DMG and MCDM workflow
DMG: prompt → provider (`task=dmg_generate`, structured `{spec_json}`) → parse → static validation (schema, fields, ops, depth/size limits, types) → LLM comprehension check (`task=dmg_validate`) → evaluate on 3 example plans → compare with expected values (tolerance 1e-6 abs/rel) → on error feed `(previous spec, error)` back; at most `max_attempts`; failure → supervisor warning, nothing is evaluated.
MCDM: feasible candidates + metric matrix + preference text → provider (`task=mcdm_select`) → output validation (id exists, id feasible, justification present, criteria known) → record. Justification is stored as `llm_explanation`; computed values as `evidence`.

## 6. Provider interface (`llm/base.py`)
`LLMProvider.generate_structured(task, system, user, json_schema) -> LLMResponse(data, raw_text, provider, model, is_mock, latency_s, input_tokens|None, output_tokens|None)`; typed errors (`LLMAuthError`, `LLMRateLimitError`, `LLMTimeoutError`, `LLMOutputError`, `LLMUnavailableError`); bounded retry with backoff on rate-limit/timeout only. `verify_model()` lists provider models and raises if `LLM_MODEL` is absent; no fallback.

## 7. Offline / demo mode
`MockProvider` (deterministic, `is_mock=True`). Every record produced with it carries `provider=mock` and the UI shows a persistent banner. Mock decisions are never labelled as LLM results.

## 8. Frontend architecture
`app.py` → `st.navigation` over `frontend/pages/*.py`; `frontend/helpers.py` (pure functions, unit-tested), `frontend/theme.py` (small CSS), `frontend/state.py` (session-state keys). No business logic in the UI. Long operations run behind an explicit button with `st.status` feedback.

## 9. Test strategy
Unit (hand-verifiable instances, DSL, DMG loop incl. failure/retry, MCDM validation, providers with fake SDK clients, config/redaction), integration (routing→DMG→MCDM with mock), regression (fixed seeds → frozen plan fingerprints), UI helper tests and a headless `streamlit.testing.v1.AppTest` smoke test. Optional live tests gated by `RUN_LIVE_LLM_TESTS=1`.

## 10. Experiment strategy
See `experiment_protocol.md`.

## 11. Risks and limitations
Synthetic instances; heuristic and prompts are reconstructions; mock provider results demonstrate mechanics only; live provider paths could not be exercised from the build environment; Windows commands not executed on Windows.

## 12. Planned outputs
`results/runs/<run_id>/` (instance, candidates, dmg_trace, mcdm_trace, record), `results/experiments/<exp>/<run_id>/` (results.csv, summary.json, meta.json), `logs/run_log.jsonl`, `logs/repair_log.md`, `docs/reproduction_report.md`.

## 13. Intentional deviations
See `uncertainties_and_deviations.md` (U01 DSL instead of generated code; U02 provider; U03–U17 reconstructions).
