# Reproduction of "Improving Adaptability in Optimization-Based Decision Support Systems Through Large Language Models"

Ghiani, Manni, Zacchino. *IEEE Access* 13:149777-149788 (2025), DOI 10.1109/ACCESS.2025.3601518.
The paper's PDF is included as `base_paper.pdf`.

This project rebuilds the paper's method and its experiments **as far as the published text allows**, and shows it in a dashboard.
It does not claim an exact reproduction. Read `docs/reproduction_report.md` for what was and was not established.

## What the software does

1. **Routing.** A multi-period dynamic vehicle-routing problem with pickups and deliveries. A cheapest-insertion heuristic builds *m* alternative plans. A separate validator checks capacity, precedence, time windows, shift length and request coverage. Only the validator marks a plan feasible.
2. **Dynamic metric generation (DMG).** A user describes a new metric in words. A language model writes a metric specification, code checks it, the model checks it again, and it is run on three hand-built examples against expected values. Failures go back to the model for up to *N* attempts (Algorithm 1 of the paper).
3. **Multi-criteria decision making (MCDM).** Given the evaluation matrix and a preference in plain language, the model picks one feasible plan and explains why. Code checks that the plan exists, is feasible and that numbers in the explanation match the matrix.
4. **Experiments C1 to C6** mirror the paper's Section IV-B tests. The human study (Section IV-C) is **NOT RUN**.

### Deliberate differences from the paper (see `docs/uncertainties_and_deviations.md`)

- The model returns a JSON expression tree, not Python code. Nothing the model writes is ever executed with `eval` or `exec` (U01).
- Gemini (default) or OpenAI through their official SDKs, or an offline mock. The paper used GPT-4o with LangChain (U02).
- The paper's instances and prompts are not published, so instances are synthetic and prompts are our reconstructions.

### Mock provider

The mock provider is a deterministic rule-based stand-in so the software runs without a key. **It is not a language model.** Everything it produces is labelled `mock` / `synthetic_demonstration`, and the dashboard refuses to compare mock results with the paper's numbers.

## Setup (Windows PowerShell) - NOT VERIFIED ON WINDOWS

The build ran on Linux only. These commands are standard but were never executed on Windows.

```powershell
cd C:\basepaper
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src;."
```

If PowerShell blocks the activation script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

### Run the dashboard

```powershell
streamlit run app.py
```

Open the address it prints (normally http://localhost:8501). The sidebar starts in offline mock mode.

Fastest demo: on **Problem configuration** press **Run full demo**. It loads a 30-request instance, sets a preference and one metric request, and runs everything. Then open Candidate plans, Dynamic metric generation and Decision explorer.

### Run the tests

```powershell
python scripts/run_tests.py
```

This runs the unit, integration, regression and UI suites and writes `logs/test_status.json`, which the dashboard shows. Equivalent: `python -m pytest tests`.

### Run an experiment or a demo from the command line

```powershell
python -m paper_replication.cli demo --provider mock
python -m paper_replication.cli run-experiment C1 --provider mock
python -m paper_replication.cli run-all --provider mock
```

Results go to `results/runs/<run_id>/` and `results/experiments/<experiment>/<run_id>/`.

### Use a real model (Gemini or OpenAI)

1. Copy `.env.example` to `.env` and fill in a key. `.env` is git-ignored. Never put a key in source code.
2. List the model ids your key can use, then set `LLM_MODEL` to one of them. There is no default model and no silent fallback.

```powershell
copy .env.example .env
notepad .env
python -m paper_replication.cli list-models --provider gemini
python -m paper_replication.cli config
```

Real runs make many calls and can cost money. The dashboard asks for confirmation before an experiment uses a live provider.
Live provider code is tested only with fake clients. **No live API call was made during the build.**

## Dashboard pages

| Page | Purpose |
|------|---------|
| Overview | Workflow, provider status, latest run, test status, documents |
| Problem configuration | One-click demo; map of all pickup and delivery points; edit or load an instance with field-level errors; preference and metric requests; run |
| Candidate plans | Comparison table, charts, routes on a planar graph (not a geographic map), constraint check, selected plan |
| Dynamic metric generation | Generated specification, every attempt, retries, a try-it box with a real oracle, a specification checker |
| Decision explorer | Choose a plan from a preference; includes the paper's own Table 9 matrix as a clearly labelled paper input |
| Experiments | Run C1 to C6, status read from disk, charts, comparison with the paper only when valid |
| Logs and reports | Documents, run log, saved runs, test status. Secrets are redacted |

Colour badges show where each fact comes from: **verified by code**, **model output**, **mock or synthetic**, **paper value**, **extension (not in the paper)**.

## Project layout

```
app.py                 dashboard entry point
frontend/              pages, charts, helpers (no business logic)
src/paper_replication/ routing, DMG, MCDM, providers, experiments (no Streamlit import)
tests/                 unit, integration, regression, ui
scripts/run_tests.py   runs all suites and records real statuses
data/                  paper reference values and sample instances
docs/                  source audit, methodology, deviations, traceability, plan, protocol, report
results/, logs/        produced artefacts (not committed)
```

## Security

- Keys are read from the environment or a local `.env` only. Logs, reports, exceptions and the UI pass through a redactor.
- `.env.example` holds placeholders only.
- Model output is parsed as data and validated. It is never executed.

## Status of evidence

See `docs/reproduction_report.md`. In short: the mechanics are implemented and tested; the language-model behaviour reported by the paper has **not** been reproduced because no live model could be called and the paper's inputs are unavailable.
