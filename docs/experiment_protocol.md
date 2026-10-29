# Experiment Protocol

Result categories (every artifact is labelled with exactly one):
1. `original_conditions` — GPT-4o + LangChain 0.3 + paper's prompts/instances: **not possible** (inputs NOT AVAILABLE).
2. `partial_reproduction` — paper's design, our reconstructed inputs, real LLM provider.
3. `synthetic_demonstration` — mock provider and/or synthetic data (mechanics only).
4. `alternative_model` — real provider other than GPT-4o (Gemini / OpenAI model ≠ paper's).
5. `exploratory` — not in the paper.

Paper reference values live in `data/paper_reference_values.json` (cited, never copied into results). Comparison tables are shown only for `partial_reproduction`/`alternative_model` runs and always carry the caveat that inputs differ.

Common settings: seed policy — instance seed `S`, policy-sampling seed `S`, prompt-set seed `S` (default `S=20251028`); `max_attempts=5`; mock provider in CI.

| ID | Source | Research question | Instance / data | Runs & seeds | Variables | Prompt conditions | Metrics | Baselines | Aggregation | Artifacts |
|----|--------|-------------------|-----------------|--------------|-----------|-------------------|---------|-----------|-------------|-----------|
| C1 | Sec. IV-B1, Table 3 | Does DMG produce correct metrics from clear prompts? | 10 seed metrics (Table 2), 3 example plans each, 30 prompts per type | 1 pass per prompt | prompt type T/TF/TFP | clear prompts (hand-written paraphrases offline; LLM paraphrases live) | generated, avg attempts, tests passed | — | counts per type | `results/experiments/C1/<run>/{results.csv,summary.json,meta.json}` |
| C2 | Sec. IV-B2, Table 5 | Does MCDM select π\* per trade-off prompt as m grows? | evaluation matrices from real heuristic plans, m=10/50/100 | 30 prompts × each m | m | 9 seed trade-offs × paraphrases = 30 | agreement with reference rule; near-optimality rank | reference rule | correct/30 | same layout under `C2` |
| C3 | Sec. IV-B3, Table 6 | Robustness of DMG to informal/unclear/ambiguous metric prompts | C1 prompts degraded (text part only for TF/TFP) | 1 pass per prompt | degradation × type | rule-based degradations offline | generated, passed | C1 | counts | `C3` |
| C4 | Sec. IV-B4, Table 7 | Effect of trade-off prompt clarity | C2 prompts, m=10 (paper does not state m — **ASSUME** 10) degraded to levels 1–3 | 30 per level | level | rule-based degradations | correct/30 | C2 m=10 | counts | `C4` |
| C5 | Sec. IV-B5, Table 8 | Token usage and time of DMG and MCDM | m = 10,50,100 (real plans), 200,500,1000 (synthetic matrices, labelled) | 3 repetitions | m | one trade-off prompt | tokens/run (provider-reported, else *unavailable*), wall-clock s, prompt chars (proxy) | — | mean | `C5` |
| C6 | Sec. IV-B6 | Are justifications consistent with computed metrics? | C2 matrices & prompts | 30 | — | — | automated grounding check of numeric claims; full text kept for manual rating | — | rate | `C6` |
| IV-C | Sec. IV-C | Human tuning effort | — | — | — | — | — | — | — | **NOT RUN** (human study) |

Commands (after setup): `python -m paper_replication.cli run-experiment C1 --provider mock`, `... run-all --provider mock`, `... demo`, `... list-models` (live), see README.
