# Paper Methodology Specification (as read from the source)

This document restates the methodology of the base paper in implementable terms. Every statement is tagged with its status from `source_audit.md`. Anything tagged **IMPL** is *our* decision where the paper is silent.

## 1. Framework (Sec. III, Fig. 1) — VERIFIED

```
supervisor ──metrics system prompts──▶ DMG ──metric code──┐
                                                          ▼
user requests ─▶ modular optimisation algorithm ─▶ m plans ─▶ evaluation matrix (m × n)
                                                          │
supervisor ──trade-off system prompts──▶ MCDM ◀───────────┘
                                          │
                              selected plan π*, justification
```
Both LLM modules pass through a *comprehension verification phase*; failures raise a warning to the supervisor.

## 2. Problem: multi-period dynamic VRP with pickups and deliveries (Sec. IV-A)

- Fleet, one depot, requests arriving over time — VERIFIED.
- Request = pickup+delivery pair, priority (normal/high), optional time window(s) — VERIFIED.
- Decision on arrival: accept/reject; if accepted assign vehicle, service day, time window, position in route — VERIFIED.
- Capacity limited — VERIFIED (values: IMPL).
- Time recursion a/s/d/w from p.149784 — VERIFIED; implemented in `constraints.py` (service duration 0 as implied; configurable field defaulting to 0).
- Travel cost/time model, geometry — **IMPL**: Euclidean distance on a planar grid; travel time = distance × `minutes_per_unit`.
- Day structure — **IMPL**: horizon of D days; "today" is the arrival day of the last processed request; a request may be scheduled on any day ≥ its arrival day within `max_lead_days`.

## 3. Cheapest-insertion heuristic — name only (PARTIALLY SPECIFIED)

IMPL (textbook sequential cheapest insertion for pickup–delivery pairs):
1. Requests are processed in arrival order (arrival day, then arrival sequence) — the dynamic setting.
2. For each request, enumerate every (day, vehicle, pickup position i, delivery position j ≥ i) and compute the added distance of inserting pickup then delivery.
3. Test hard feasibility (capacity along the route, pickup before delivery, time windows, shift duration, depot closing time) with the deterministic validator.
4. Insert at the feasible position with minimum *policy-adjusted* insertion cost; ties broken by (day, vehicle id, i, j) lexicographically.
5. If no feasible insertion exists, or a policy rule rejects the request, it is rejected. Accepted requests are never re-optimised.
6. The m alternative plans (paper: "returns m alternative solutions") are obtained by running the same heuristic under m different *policy parameter vectors* (day-preference penalty, detour cap, tight-window rejection, shift limit, reserved capacity for high priority, peak-hour avoidance, balance penalty). This is IMPL; the paper does not say how its alternatives were produced.

## 4. DMG (Sec. III-A, III-C, Algorithm 1) — VERIFIED structure

Input: `prompt = {metric, examples, expected_results}`, `max_attempts`.

```
attempts ← 0; code ← ""; error ← ""
while attempts < max_attempts:                     # text: "at most max_attempts iterations"
    code ← generate(prompt.metric, code, error)    # LLM; self-debug uses previous code+error
    v ← validate(prompt.metric, code)              # LLM comprehension check (+ static checks IMPL)
    if v is Success:
        r ← execute(code, prompt.examples)
        if r == prompt.expected_results: return ("Success", code)
        error ← r.error
    else: error ← v.error
    attempts ← attempts + 1
return ("Failure", error)  → warning to supervisor
```
- **Deviation (required by the project brief and safety):** the paper generates *executable code*. We generate a *constrained metric specification* (JSON expression tree, DSL) that is statically validated and interpreted by our own evaluator — no `eval`/`exec`. The loop, verification phases, examples/expected results and bounded self-debugging are otherwise as in the paper.
- Prompt types T / TF / TFP — VERIFIED; prompt wording — IMPL (the paper's prompts are NOT AVAILABLE).
- `max_attempts` default — IMPL (5).

## 5. MCDM (Sec. III-B, III-D) — PARTIALLY SPECIFIED

- Input: evaluation matrix (candidate id × metric values) and a natural-language trade-off prompt.
- Output: selected candidate id π\* and a natural-language justification — VERIFIED.
- IMPL: structured output `{selected_plan_id, justification, criteria_used}`; the id must exist among *feasible* candidates; invented ids are rejected; the justification is stored separately from computed metrics; numeric claims in the justification can be checked against the matrix (grounding check).
- Ground truth for evaluating selection accuracy (paper: NOT SPECIFIED) — IMPL: for every seed trade-off prompt we define a deterministic reference scoring rule (primary / secondary criteria and tolerance) in `mcdm/reference_rules.py`. Reported accuracy is "agreement with this reference rule" and is therefore only comparable to the paper's Table 5 qualitatively.

## 6. Metrics (Table 2) — text VERIFIED; exact formulae IMPL where marked

| id | description (paper) | our formalisation |
|----|--------------------|-------------------|
| z1 | fraction of accepted requests under the current plan | accepted / received |
| z2 | fraction of accepted requests on the current day ("today") | **IMPL:** accepted / received among requests that arrived today |
| z3 | total distance travelled by the fleet | sum over routes of route distance |
| z4 | total duration of vehicle routes | sum of (end − departure) |
| z5 | total waiting time | sum of w = max(0, e − a) over stops (paper formula) |
| z6 | mean absolute deviation of route durations today | MAD of durations of today's non-empty routes |
| z7 | % deviation of average route durations "today" vs "tomorrow" | 100·|avg_today − avg_tomorrow| / avg_today (**IMPL**; paper gives no formula) |
| z8 | number of customers in a specific ZIP code visited at specific times (peak) | stops in target ZIP with service start within peak window (**IMPL**: ZIP and window are parameters) |
| z9 | weighted score over accepted requests: 10 today, 5 future date, 0 rejected | **IMPL:** 10 if scheduled day ≤ today, 5 if later, 0 rejected |
| z10 | weighted score priority vs non-priority: 10 scheduled priority, 5 scheduled non-priority | **IMPL:** rejected = 0 |

## 7. Experiments

See `experiment_protocol.md`; all of Tests C1–C6 follow the paper's design where the paper gives one and are labelled approximations otherwise.
