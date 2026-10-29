# Uncertainties, Assumptions and Deviations

Categories: **SRC** (source-verified fact), **IMPL** (implementation decision where paper is silent), **ASSUME** (assumption), **DEV** (intentional deviation), **NA** (evidence unavailable).

| ID | Cat | Topic | Paper says | What this project does | Consequence |
|----|-----|-------|-----------|------------------------|-------------|
| U01 | DEV | DMG output | LLM generates executable code | LLM emits a JSON metric specification (expression-tree DSL) validated and interpreted by our evaluator; no `eval`/`exec` | Table 3/6 analogues are *not* directly comparable to the paper's code-generation accuracy |
| U02 | DEV | Runtime LLM | GPT-4o via LangChain 0.3 | Gemini (default) / OpenAI via official SDKs, or deterministic mock; no LangChain | Any live result is an *alternative-model comparison*, never "the same configuration" |
| U03 | IMPL | Heuristic | "simple cheapest insertion" | Sequential cheapest insertion over (day, vehicle, i, j) with deterministic tie-breaks | Plans differ from the paper's; Table 9 values cannot be reproduced |
| U04 | IMPL | m alternative plans | Not stated | m policy-parameter variants of the heuristic | Candidate diversity is our design |
| U05 | IMPL | Instance generator | Not stated | Seeded synthetic generator (planar grid, ZIP zones, priorities, time windows, arrival days) | All results are on **synthetic** instances |
| U06 | IMPL | Travel model | Not stated | Euclidean distance; minutes per unit configurable (default 1.0) | |
| U07 | IMPL | Service time | formula d_j = s_j (zero) | `service_time` per request, default 0 | Matches paper formula by default |
| U08 | IMPL | Departure time | Not stated | Vehicle departs at max(shift_start, first_service_start − travel) to avoid artificial initial waiting | Waiting time z5 counts only waiting between stops |
| U09 | IMPL | "Today" | "current day" | Arrival day of the last processed request | z2, z6, z7, z9 depend on this |
| U10 | IMPL | z2, z7, z8, z9, z10 formulae | verbal only | see `paper_methodology.md` §6 | |
| U11 | IMPL | Algorithm 1 bound | pseudocode `≤`, text "at most max_attempts" | at most `max_attempts` passes (text reading) | |
| U12 | IMPL | LLM comprehension validation | LLM validates code vs description | LLM validates spec vs description (provider call) **and** deterministic static checks; hard constraints are never delegated to the LLM | Adds safety, not in paper |
| U13 | IMPL | Prompts (metric, trade-off, validation, justification) | NA | Own templates (`dmg/generation.py`, `mcdm/prompts.py`) | |
| U14 | IMPL | Paraphrases/degradations | LLM-generated | Offline: hand-written template paraphrases and rule-based degradations (labelled). Live: optional LLM paraphrasing when a real provider is configured | Offline prompt sets are *not* the paper's prompt sets |
| U15 | IMPL | MCDM ground truth | NA | Deterministic reference rules per seed trade-off prompt (`mcdm/reference_rules.py`) | "Accuracy" = agreement with our rule |
| U16 | IMPL | Table 4 numbering | prompts 1,2,3,5–10 (no #4) | Keep paper numbering; 9 seed prompts; no invented #4 | |
| U17 | IMPL | Evaluation matrices for m>100 | NA | m ≤ 100 from real heuristic plans; larger m (C5 only) from bootstrap-with-jitter *synthetic* matrices, labelled | C5 token/time not comparable to Table 8 |
| U18 | NA | Token/latency under mock | — | Reported as *unavailable* for tokens; wall-clock of mock is a mechanics check only | |
| U19 | NA | IV-C human study | 10 humans, 19.8 attempts | **NOT RUN** (human participants required). Paper value stored as reference only | |
| U20 | NA | C6 qualitative judgement | authors' judgement | Automated *grounding check* of numeric claims + stored text for manual review; not equivalent | |
| U21 | ASSUME | Model identifiers | GPT-4o | Cannot be verified from the offline build environment (API hosts unreachable). `list-models` CLI verifies against the provider at run time; no silent fallback | Live provider path is **BLOCKED** here (see report) |
| U22 | IMPL | Structured output | NA | Provider-native JSON-schema output with a flat envelope `{spec_json: string}`; recursion avoided for cross-provider compatibility | |
| U23 | SRC | Inconsistencies in paper | — | Not "corrected" silently: Table 4 gap, TFP/TP, z8/z9 mix-up in π₃ example, C4 definitions | |
