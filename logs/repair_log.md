# Repair log

Rule: at most 3 repair attempts per distinct failure. Entries state the evidence, the cause and whether the *test* or the *code* was wrong.

## R-001 tests/unit/test_insertion.py::test_multi_vehicle_use_when_capacity_forces_it
- Evidence: assertion `{'V1'} == {'V1','V2'}`.
- Cause: the **test** was wrong. Capacity 2 with demand 2 does not force a second vehicle because one vehicle can deliver each request before picking up the next.
- Attempt 1: rewrote the scenario so the 60 minute shift is what forces a second vehicle. Production code unchanged.
- Attempt 2: the rewritten scenario was still wrong. Identical coordinates cost nothing extra, so one vehicle serves all four requests in one trip. Production code was correct (plan feasible, all accepted). The test now uses east and west requests whose combined route exceeds the shift.

## R-002 tests/unit/test_dmg.py::test_mock_provider_generates_reference_metric_for_clear_prompt
- Evidence: DMG failed after 5 attempts.
- Cause: the **test** was wrong. It asked for z3 with the text 'accepted over received', which the keyword-based mock correctly ranked as z1 and the oracle (z3 values) rejected. That is the oracle doing its job.
- Attempt 1: the test now uses the Table 2 description of z3. Production code unchanged.

## R-003 tests/unit/test_mcdm.py::test_reference_rules_on_table9_matrix (also a real UI defect)
- Evidence: `KeyError: 'z11'`.
- Cause: **code** defect in the Decision explorer page, not just the test. Reference rules 3, 8, 9, 10 use extension metrics z11 to z14, which the paper's Table 9 matrix does not contain.
- Attempt 1: the page now shows an explanatory note instead of calling the rule when its metrics are absent; the test asserts exactly which rules are usable on Table 9 (1, 2, 5, 6, 7).

## R-004 tests: test_candidates, test_prompt_sets, test_ui_helpers (3 test defects)
- Evidence: `produced_m == 10` for the tiny instance; `TradeoffPrompt` has no attribute `degradation`; pandas refused to put 'many' in an int64 column.
- Cause: all three were **test** defects (wrong assumption about how few plans the tiny instance has, wrong field name `degradation_level`, strict pandas dtype). Production code unchanged. Attempt 1 fixed each.

## R-005 tests/unit/test_metrics_reference.py::test_examples_distinguish_metrics (real defect in the DMG examples)
- Evidence: z11 equals 1.0 on all three hand-built examples, so an incorrect z11 implementation could pass the example test.
- Cause: **code/data** defect. Every example contained exactly one tight-window request.
- Attempt 1: example 1 now widens request c's window to 120 minutes (z11 becomes 0 there). Other metric values on that example are unchanged because the vehicle still departs for the window's opening time.

## R-006 tests/integration/test_experiments.py::test_c4_c5_c6_small (real defect)
- Evidence: C5 summary keys were '5.0' and '10.0'; the paper-comparison lookup uses '10', '50', '100', so it would have returned nothing silently.
- Cause: **code** defect. The `m` column becomes float because DMG rows have `None`.
- Attempt 1: key by `str(int(m))`. While fixing this I also made `mean_or_none` / `sum_or_none` treat NaN as unavailable, because pandas converts `None` to NaN and the old code could have reported a NaN token mean for the mock provider. Added unit and integration tests for both.

## R-007 Overview page showed '0 passed, 0 failed' (found by looking at a real browser screenshot)
- Evidence: screenshot of the running app; `logs/test_status.json` has per-suite counts but the page read top-level keys that do not exist.
- Cause: **code** defect in `frontend/pages/overview.py` (schema mismatch). The headless AppTest did not catch it because nothing raised.
- Attempt 1: added `helpers.test_totals` (summing suites) with a unit test and used it on the page.
