# Source Audit — Ghiani, Manni, Zacchino (IEEE Access 13, 2025)

**Paper:** "Improving Adaptability in Optimization-Based Decision Support Systems Through Large Language Models", IEEE Access, vol. 13, pp. 149777–149788, 2025. DOI 10.1109/ACCESS.2025.3601518 (the DOI printed on PDF page 1 is `10.1109/ACCESS.2025.3601518`; title and page range match).
**File audited:** `base_paper.pdf` (12 PDF pages = journal pages 149777–149788; PDF page *p* = journal page 149776 + *p*).
**Audit date:** 2026-10-28. Tables 2–9 were verified by rasterising the PDF pages and reading the images (the tables are embedded as images, so text extraction does not return them).

Status vocabulary: **VERIFIED**, **PARTIALLY SPECIFIED**, **NOT SPECIFIED**, **NOT AVAILABLE**.
References use journal page numbers (PDF page in brackets).

---

## A. Problem formulation and research objective

| # | Item | Source | Status |
|---|------|--------|--------|
| A1 | Objective: add flexibility/adaptability to optimization-based DSSs by using LLMs to (a) interpret natural-language instructions, (b) dynamically reconfigure algorithmic components, (c) support preference-based decision making, (d) explain the rationale of choices. | p.149778 [2], Sec. I; p.149780–81 [4–5], Sec. III | VERIFIED |
| A2 | Architecture (Fig. 1): user requests → modular optimisation algorithm → evaluation matrix (m plans × n metrics); DMG module driven by *metrics system prompts*; MCDM module driven by *trade-off system prompts*; *comprehension verification* with warning to the *supervisor*; outputs selected plan π\* and decision justification. | p.149780 [4], Fig. 1; Sec. III-A..D | VERIFIED |
| A3 | Hypothesis: LLMs + classical optimisation add flexibility without sacrificing solution quality or computational efficiency. | p.149782 [6], Sec. IV intro | VERIFIED |
| A4 | Six evaluation criteria C1–C6 (DMG correctness; MCDM correctness; robustness to ill-defined metrics; robustness of MCDM to contradictions/inconsistencies; computational efficiency; interpretability). | p.149783 [7], Sec. IV-B | VERIFIED |
| A5 | **C4 as defined** ("detect contradictions and inconsistencies in company policies") differs from **Test C4 as run** (progressively reducing prompt clarity, Table 7). | p.149783 [7] vs p.149784–85 [8–9] | VERIFIED (internal inconsistency in paper) |

## B. The multi-period dynamic VRP with pickups and deliveries

| # | Item | Source | Status |
|---|------|--------|--------|
| B1 | Problem name: *Multi-Period Dynamic Vehicle Routing Problem with Pick-ups and Deliveries*. | p.149782 [6], IV-A | VERIFIED |
| B2 | Fleet of vehicles operating from a **central depot**. | p.149782 [6] | VERIFIED |
| B3 | Pickup and delivery requests **arrive dynamically over time**. | p.149782 [6] | VERIFIED |
| B4 | Request attributes: **priority level** (normal / high) and, **optionally, time windows** for pickup and/or delivery. | p.149782 [6] | VERIFIED |
| B5 | On receipt the company decides **accept / reject**. If accepted: assigned to a **vehicle**, a **service day**, a **time window**, and **sequenced** (position in that vehicle's route for that day). | p.149782 [6] | VERIFIED |
| B6 | **Limited fleet capacity**; accepting a request may preclude future ones. | p.149782 [6] | VERIFIED (capacity values NOT SPECIFIED) |
| B7 | Time recursion: a_j = d_i + t_ij; s_j = max(e_j, a_j); w_j = max(0, e_j − a_j); d_j = s_j (e_j,l_j time window; a,s,d,w arrival, service-start, departure, waiting). Implies zero service duration. | p.149784 [8], IV-B1 | VERIFIED (formulae given in the discussion of z5 failures) |
| B8 | Real-world extras mentioned as possible objectives/constraints: maximise accepted requests; prefer same-day service even at longer routes; accept early servicing only within tolerable detour; reject tightly constrained requests; avoid congested ZIP zones at peak hours; workload balance; reserve capacity for expected high-priority requests; vehicles returning after a threshold hour (e.g. 18:00); limit long routes on consecutive days; customer–driver continuity. | p.149782–83 [6–7] | PARTIALLY SPECIFIED — listed as motivations; **not** stated as hard constraints of the prototype. They enter the study only through metrics (Table 2) and trade-off prompts (Table 4). |
| B9 | Adjustable DSS parameters of the baseline DSS mentioned in IV-C: driver shift durations (e.g. 8 h → 7 h), customer time windows, customer demands. | p.149786 [10], IV-C | PARTIALLY SPECIFIED |
| B10 | Fleet size, capacity, number of days/periods, number of requests, geography, travel-time model, service times, shift length, depot hours, request-arrival process, priority mix, time-window generation. | — | NOT SPECIFIED |
| B11 | Distance units/scale: Table 9 reports total distances z3 ≈ 2115–2430 (units not stated). | p.149786 [10], Table 9 | PARTIALLY SPECIFIED |

## C. Cheapest-insertion heuristic

| # | Item | Source | Status |
|---|------|--------|--------|
| C1 | Prototype uses "a **simple cheapest-insertion heuristic**" for the multi-period dynamic VRPPD. | p.149783 [7], IV-A (last paragraph) | VERIFIED (name only) |
| C2 | The heuristic "returns **m alternative solutions**, each evaluated w.r.t. the customer-defined metrics." | p.149783 [7] | VERIFIED (mechanism for producing m plans NOT SPECIFIED) |
| C3 | Initialisation, insertion order, insertion-cost definition, feasibility checks, tie-breaking, day-selection rule, accept/reject rule, how the m variants differ (parameters? randomisation?). | — | NOT SPECIFIED |

→ The exact heuristic cannot be reproduced from the paper. We implement the textbook sequential cheapest insertion for pickup–delivery pairs and document every decision (see `uncertainties_and_deviations.md`).

## D. Dynamic Metric Generation (DMG)

| # | Item | Source | Status |
|---|------|--------|--------|
| D1 | DMG is the module generating **code** for components not predefined (new metrics), guided by *metrics system prompts* from the human supervisor. | p.149781 [5], III-A | VERIFIED |
| D2 | Comprehension verification: (a) the generated code is validated **by an LLM** against the prompt's description; (b) it is **executed on representative examples** and results compared with expected values; (c) on failure (validation failure, run-time error, mismatch) the module **self-debugs** using the previous errors, for at most `max_attempts` iterations. | p.149781 [5], III-C | VERIFIED |
| D3 | If it fails, the system **issues a warning to the supervisor**. | p.149781 [5] | VERIFIED |
| D4 | Algorithm 1 (pseudocode): input prompt, `max_attempts`; loop `while attempts ≤ max_attempts`; `generate(metric, code, error)`; `validate(metric, code)`; `execute(code, examples)`; compare with `expected_results`; return ("Success", code) or ("Failure", error). | p.149782 [6], Algorithm 1 | VERIFIED — note: pseudocode uses `≤` with attempts starting at 0 (max_attempts+1 passes) whereas the text says "at most max_attempts iterations". |
| D5 | Prompt object has fields `metric`, `examples`, `expected_results`; **three examples** per metric are used. | p.149781 [5]; p.149783 [7] | VERIFIED |
| D6 | Prompt types: T (text), TF (text+formulae, LaTeX), TFP (text+formulae+pseudocode, LaTeX). Text calls the third type both "TFP-prompts" and "TP-prompts". | p.149783 [7]; p.149785 [9] | VERIFIED |
| D7 | Output representation: "code snippets"; language, library API, function signature, data format of "examples" and of plans. | — | NOT SPECIFIED |
| D8 | Validation prompt text, generation prompt text, self-debug prompt text, `max_attempts` value, temperature/decoding settings. | — | NOT AVAILABLE |
| D9 | Hard constraints are not delegated to the LLM — **not stated** by paper (it concerns metrics only). | — | NOT SPECIFIED |

## E. Multi-Criteria Decision Making (MCDM)

| # | Item | Source | Status |
|---|------|--------|--------|
| E1 | m plans π₁…π_m with n features z₁…z_n form the **evaluation matrix**; trade-off system prompts convey relative importance, acceptable compromises, supervisor priorities. | p.149781 [5], III-B | VERIFIED |
| E2 | The MCDM module applies a "structured decision-making methodology (weighted scoring, utility theory, outranking)" to select π\*; the paper does not state that any of these is implemented — selection is performed by the LLM. | p.149781 [5]; p.149784 [8] (tie behaviour) | PARTIALLY SPECIFIED |
| E3 | Comprehension verification is also applied to MCDM, but only the DMG procedure is described in detail. | p.149781 [5], III-C | PARTIALLY SPECIFIED |
| E4 | Justification: LLM produces a natural-language rationale from evaluation matrix + trade-off prompt. Example for π₃ with trade-off prompt 3 and Table 9. | p.149781–82 [5–6], III-D; p.149785–86 [9–10], IV-B6 | VERIFIED |
| E5 | Seed trade-off prompts: Table 4 (**9 prompts, numbered 1,2,3,5,6,7,8,9,10 — #4 is absent**). | p.149784 [8], Table 4 | VERIFIED (numbering gap) |
| E6 | Paraphrase protocol: an LLM generates paraphrased variations; test set of **30 distinct prompts**. Whether 30 is per seed or total is not stated; Table 5 reports out of 30, so total is the consistent reading. | p.149784 [8], IV-B2; Table 5 | PARTIALLY SPECIFIED |
| E7 | Evaluation matrices of sizes m = 10, 50, 100 for Test C2; m up to 1000 for Test C5. How matrices of size ≥ 100 were produced (heuristic outputs vs synthetic) not stated. | p.149784 [8]; p.149785 [9] | NOT SPECIFIED |
| E8 | Ground truth for "correct" π\* (what makes a selection correct; tolerance when no plan is optimal for all criteria). The text says that when π\* is not optimal it "remains close to the optimum". | p.149784 [8] | NOT SPECIFIED |
| E9 | Tie handling: when plans tie on the prompt's measures, the LLM selects the plan that performs best on the other measures via common-sense reasoning. | p.149784 [8]; p.149786 [10] | VERIFIED (observation, not a rule) |
| E10 | Structured output format for the decision; reference to candidate IDs. | — | NOT SPECIFIED |

## F. Runtime LLM and framework

| # | Item | Source | Status |
|---|------|--------|--------|
| F1 | LLM: "ChatGTP-4o [sic]" — i.e. GPT-4o; reference [17] cites OpenAI GPT-4 page. | p.149783 [7]; ref. [17] p.149787 [11] | PARTIALLY SPECIFIED (no snapshot/version date, no decoding parameters) |
| F2 | Framework: LangChain 0.3. | p.149783 [7]; ref. [40] | VERIFIED |
| F3 | Prompt templates, chains/agents used, structured-output mechanism. | — | NOT AVAILABLE |

## G. Experimental set-up and reported results

| # | Item | Source | Status |
|---|------|--------|--------|
| G1 | Metrics z1…z10 (Table 2). | p.149783 [7], Table 2 | VERIFIED (text); precise formulae for z1, z2, z8, z9, z10 are only given verbally → PARTIALLY SPECIFIED |
| G2 | **C1 / Table 3** — 3 prompt types × (30 prompts); Generated 27/25/27, avg attempts 1.42/1.58/1.42, tests passed 27/25/27. Failures: T-prompts for z5; 5 failures for TF, 3 for TFP. | p.149783–84 [7–8] | VERIFIED |
| G3 | **C2 / Table 5** — correct selections out of 30: m=10 → 30; m=50 → 28; m=100 → 25. | p.149784–85 [8–9] | VERIFIED |
| G4 | **C3 / Table 6** — informal/unclear/ambiguous degradation of C1 prompts (only the text part is degraded for TF/TFP). Generated/Passed: T 21/21, 18/15, 18/15; TF 18/18, 18/18, 21/21; TFP 24/24 ×3. Why the "Generated" counts are 18/21/24 (not 30) is not explained. | p.149785 [9] | VERIFIED (denominators NOT SPECIFIED) |
| G5 | **C4 / Table 7** — trade-off prompt degradation levels 1,2,3: 30, 30, 24 correct. | p.149785 [9] | VERIFIED |
| G6 | **C5 / Table 8** — tokens/run and time: DMG code generation 7,114 tok / 3.75 s; DMG validation 6,254 / 0.99 s; MCDM m=10 1,713 / 4.27 s; 50: 5,537 / 6.12; 100: 10,107 / 7.16; 200: 21,882 / 10.27; 500: 54,772 / 12.42; 1000: 109,736 / 15.09. Hardware/network conditions not stated. | p.149785 [9] | VERIFIED |
| G7 | **C6** — qualitative judgement by the authors that all justifications were satisfactory; example for π₃. No rubric. | p.149785–86 [9–10] | VERIFIED (qualitative; rubric NOT SPECIFIED) |
| G8 | **Table 9** — example evaluation matrix (10 plans × z1…z10). | p.149786 [10] | VERIFIED |
| G9 | **IV-C human comparison** — 10 individuals not involved in the research acted as decision makers adjusting driver shift durations, time windows and demands; average 19.8 attempts; wide variability (Fig. 2 box+violin plot). Authors explicitly say the figures are not generalisable. | p.149786 [10], Fig. 2 | VERIFIED (raw data NOT AVAILABLE; human study — cannot be reproduced) |
| G10 | Dataset / instance counts, seeds, repetitions of the VRP experiments. | — | NOT SPECIFIED |
| G11 | Baselines: only the human-in-the-loop parameter tuning of IV-C. No algorithmic baseline for DMG or MCDM. | p.149786 [10] | VERIFIED |
| G12 | Cost measurements (USD): none reported. | — | NOT SPECIFIED |

## H. Reproducibility gaps

- **NOT AVAILABLE:** source code; datasets/instances; all prompts (system prompts, paraphrases, degradations); LLM snapshot and decoding parameters; the LLM-generated paraphrases and degraded prompts; raw logs; human-study data.
- **NOT SPECIFIED:** the heuristic's details; how m alternative plans were produced; ground-truth rule for MCDM correctness; denominators in Table 6; hardware for Table 8.
- **Cannot be independently reproduced:** Table 3/5/6/7 values *as measured with GPT-4o* (model access, prompts and instances unavailable); Table 8 absolute timings (hardware/network dependent); IV-C human study; C6 qualitative judgement.
- **Internal inconsistencies noted:** Table 4 numbering skips #4; "TFP" vs "TP"; Algorithm 1 loop bound vs text; the π₃ justification example cites "z8 = 235" where Table 9 shows that value under z9; "ChatGTP-4o" typo; C4 definition vs Test C4.

## What this implies for the project

1. Architecture, metric definitions, DMG verification loop, MCDM workflow, prompt types and test designs can be implemented faithfully.
2. The VRP heuristic, instance generator, prompt texts, ground-truth rule and all LLM measurements are **our own reconstructions** and are labelled as such.
3. Paper numbers are kept in `data/paper_reference_values.json`, separate from anything our code produces, and are never presented as our results.
