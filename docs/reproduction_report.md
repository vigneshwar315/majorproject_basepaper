# Reproduction Report

**Paper:** Ghiani, Manni, Zacchino. "Improving Adaptability in Optimization-Based Decision Support Systems Through Large Language Models". *IEEE Access* 13:149777-149788 (2025). DOI 10.1109/ACCESS.2025.3601518.
**Build date:** 2026-10-28. **Build environment:** Linux container, Python 3.13. **Windows:** not executed.

## Verdict

**EXACT REPRODUCTION NOT ESTABLISHED.**

The method is fully implemented and its mechanics are tested: the routing model, the heuristic, Algorithm 1 (DMG), the MCDM step, all six Section IV-B experiments and the dashboard. What is missing is the evidence that would let us compare against the paper's numbers:

1. **No live language model was called.** The build environment cannot reach the Gemini or OpenAI APIs (proxy returns HTTP 403). All experiment results below come from the **mock provider**, which is a rule-based stand-in, not an LLM.
2. **The paper's inputs are not published.** Its instances, prompts, paraphrases, degraded prompts, ground-truth decisions and the exact GPT-4o snapshot are unavailable (`source_audit.md`, section H).
3. **One intentional deviation changes what is measured.** The model returns a JSON metric specification instead of Python code (U01), so a "generated metric" here is not the same object as in the paper.

With a real provider key, the same code produces `alternative_model` results (Gemini, or an OpenAI model other than GPT-4o), or `partial_reproduction` results (OpenAI GPT-4o). Those runs could support a **PARTIAL REPRODUCTION** verdict. They have not been made.

## What was verified, and how

| Claim | Evidence | Status |
|-------|----------|--------|
| Timing recursion a/s/w/d (Sec. IV-B1) matches hand calculation | `tests/unit/test_constraints.py` (single route 20 units / 500 min; 190 min waiting case; late departure for a window) | PASSED |
| Capacity, precedence, time windows, shift and depot-close limits, coverage, lead days are enforced | `test_constraints.py`, `test_insertion.py` | PASSED |
| Only the validator sets `feasible` | `test_constraints.py::test_feasible_flag_only_set_by_finalize`, integration test | PASSED |
| Cheapest insertion: hand-checked cost matrix, determinism, rejections with reasons, multi-vehicle use | `test_insertion.py` | PASSED |
| m distinct validated candidate plans | `test_candidates.py` | PASSED |
| Reference metrics z1-z14 on hand-built example 3 (z1 = 0.8, z2 = 2/3, z3 = 172.882, z5 = 190, z10 = 25) | `test_metrics_reference.py` | PASSED |
| The 3 DMG examples separate every metric (no metric has the same value on all three) | `test_metrics_reference.py::test_examples_distinguish_metrics` | PASSED (after repair R-005) |
| DSL rejects unknown ops, fields, nested aggregations, oversize trees; never executes Python | `test_dsl.py` | PASSED |
| Algorithm 1: success, static-error retry with feedback, value-mismatch retry, LLM mismatch retry, max-attempts failure with supervisor warning, fatal-error abort | `test_dmg.py` | PASSED |
| MCDM: infeasible plans hidden from the model, invented ids rejected and retried, grounding check of numeric claims | `test_mcdm.py` | PASSED |
| Providers: JSON-schema output, temperature only when set, bounded retry on transient errors only, no retry on auth errors, no key in errors, model verification without fallback | `test_providers.py` (fake SDK clients) | PASSED |
| Configuration and redaction: no default model, no silent fallback, keys absent from repr, logs and UI text | `test_config.py`, `test_ui_helpers.py` | PASSED |
| End-to-end pipeline with the mock; matrix values equal the reference metrics | `tests/integration/test_pipeline.py` | PASSED |
| Experiments C1-C6 run and write artefacts with correct status files | `tests/integration/test_experiments.py` | PASSED |
| Frozen plan fingerprints and metric values (detects unintended change; not ground truth) | `tests/regression/test_regression.py` | PASSED |
| Every dashboard page renders with and without a result | `tests/ui/test_app_smoke.py` (Streamlit AppTest) | PASSED |
| Dashboard works in a real browser, end to end (load instance, set preference, run, inspect) | Playwright screenshots in `docs/screenshots/` | Checked manually |
| Live Gemini/OpenAI calls | none possible | **BLOCKED** |
| Windows PowerShell setup | not executed on Windows | **NOT RUN** |
| Human tuning study (Sec. IV-C) | needs human participants | **NOT RUN** |

Test totals come from `logs/test_status.json`, written by `scripts/run_tests.py`. The latest totals are in the execution log. Lint: `ruff check .` reports no problems.

## Experiment results (mock provider, category `synthetic_demonstration`)

These numbers describe the **software**, not language-model quality. They are **not** compared with the paper. The paper's values are shown in separate tables, for reference only.

Run from `python -m paper_replication.cli run-all --provider mock`, seed 20251028, total time about 42 s. Artefacts: `results/experiments/<C*>/<run_id>/`.

### C1: DMG on clear prompts (30 prompts per type)

| Type | Reached execution | Passed example tests | Mean attempts |
|------|------------------:|---------------------:|--------------:|
| T | 30 | 29 | 1.0 |
| TF | 30 | 29 | 1.0 |
| TFP | 30 | 29 | 1.0 |

The one failure per type is a z6 prompt: the mock's keyword scoring does not map that paraphrase to z6, and the example oracle rejects every wrong guess. This shows the oracle working. It says nothing about GPT-4o.

### C2: MCDM agreement with the reference rules (30 prompts x 3 matrices from real heuristic plans)

| m | Correct (mean of 30) | Invalid outputs | Mean z1 gap of wrong picks |
|---|---------------------:|----------------:|---------------------------:|
| 10 | 22.3 | 0 | 0.061 |
| 50 | 19.0 | 0 | 0.054 |
| 100 | 17.7 | 0 | 0.057 |

"Correct" means agreement with our reconstructed lexicographic rules (U15), not with the authors' judgement.

### C3: DMG with degraded metric prompts (passed of 30; reached execution in brackets)

| Type | Informal | Unclear | Ambiguous |
|------|---------:|--------:|----------:|
| T | 29 (30) | 16 (26) | 19 (29) |
| TF | 29 (30) | 27 (30) | 29 (30) |
| TFP | 29 (30) | 27 (30) | 29 (30) |

Degradations are rule-based rewrites of the text part only (U14). The pattern that formulas and pseudocode protect against unclear text is visible, but it comes from the mock's keyword scoring.

### C4: MCDM with degraded trade-off prompts (m = 10 assumed, 3 matrices)

| Level | Correct (mean of 30) |
|-------|---------------------:|
| 1 | 22.3 |
| 2 | 28.0 |
| 3 | 29.0 |

Accuracy *rising* with degradation is a mock artefact: removing words removes keywords that pull the weighted sum away from z1. A real model is needed for a meaningful result.

### C5: Cost versus m

The mock reports no tokens, so token counts are **unavailable** (stored as `null`, never 0). Prompt size grows roughly linearly with m: about 3.7k characters at m = 10, 12.1k at m = 100 and 118k at m = 1000. Matrices for m >= 200 are **synthetic** bootstrap matrices (U17). Wall-clock times are milliseconds and only check the mechanics.

### C6: Justification grounding

30 of 30 decisions were valid and every numeric claim matched the matrix. This is expected: the mock writes the real numbers into its text by construction. The paper's C6 is a qualitative judgement by the authors; here the full texts are stored in `results.csv` for manual rating.

### Paper-reported values (reference only, from `data/paper_reference_values.json`)

| Table | Values |
|-------|--------|
| Table 3 (C1) | T: 27 generated, 1.42 attempts, 27 passed. TF: 25, 1.58, 25. TFP: 27, 1.42, 27 |
| Table 5 (C2) | m = 10: 30/30, m = 50: 28/30, m = 100: 25/30 |
| Tables 6-8 | See the JSON file (page citations included) |

## Defects found by testing and fixed

Full details in `logs/repair_log.md`. Each failure was fixed within the 3-attempt limit.

- **R-003.** The Decision explorer crashed on the paper's Table 9 matrix for rules 3, 8, 9 and 10, because they need extension metrics z11-z14 that Table 9 lacks. The page now explains this instead.
- **R-005.** Every DMG example had exactly one tight-window request, so a wrong z11 could pass. Example 1 was changed so z11 differs.
- **R-006.** C5 summary keys came out as `'10.0'`, which silently broke the paper-comparison lookup. Token means could also become NaN instead of "unavailable". Both are fixed.
- **R-007.** The Overview page showed "0 passed" because it read the test-status file with the wrong schema. This was found from a browser screenshot.
- R-001, R-002 and R-004 were defects in the tests themselves; production code was correct.

## Unresolved issues and limitations

- Live provider adapters are unit-tested only with fake clients. The first live run may surface SDK or model differences. Run `python -m paper_replication.cli list-models` first.
- Windows setup commands in the README are standard but unverified.
- The heuristic, the m-plan diversity mechanism, the instance generator, the metric formulas for z2 and z7-z10, the prompts and the MCDM ground-truth rules are reconstructions (U03-U15).
- The paper's internal inconsistencies are recorded, not corrected: the Table 4 numbering gap, the TFP/TP naming, the z8/z9 mix-up in the pi3 example, and the C4 definition (U23).

## How to move to a partial reproduction

1. Put a key in `.env`, set `LLM_PROVIDER` and a verified `LLM_MODEL`.
2. `python -m paper_replication.cli run-all` (at least about 1,300 model calls in total, more when DMG retries; see the Experiments page for per-test estimates).
3. The dashboard then shows the comparison with the paper, labelled as indicative, because inputs still differ.
