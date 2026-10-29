"""Run every test suite and write logs/test_status.json (the only source of the test status shown in the dashboard).

Status words: PASSED, FAILED, SKIPPED (all tests skipped), BLOCKED (could not run), NOT RUN.
Usage (from the project folder):  python scripts/run_tests.py [--live]
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ["unit", "integration", "regression", "ui"]


def run_suite(name: str, env: dict) -> dict:
    xml = ROOT / "logs" / f"junit_{name}.xml"
    xml.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", f"tests/{name}", "-q", f"--junitxml={xml}", "-p", "no:cacheprovider"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    except Exception as exc:  # noqa: BLE001
        return {"status": "BLOCKED", "detail": f"{type(exc).__name__}: {exc}", "tests": 0, "passed": 0, "failed": 0, "skipped": 0}
    try:
        suite = ET.parse(xml).getroot()
        suite = suite if suite.tag == "testsuite" else suite[0]
        tests, fails = int(suite.get("tests", 0)), int(suite.get("failures", 0)) + int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
    except Exception:  # noqa: BLE001
        return {"status": "BLOCKED", "detail": "pytest produced no result file", "tests": 0, "passed": 0, "failed": 0, "skipped": 0, "output_tail": proc.stdout[-800:] + proc.stderr[-800:]}
    passed = tests - fails - skipped
    status = "FAILED" if fails else ("SKIPPED" if tests and skipped == tests else ("PASSED" if tests else "NOT RUN"))
    return {"status": status, "tests": tests, "passed": passed, "failed": fails, "skipped": skipped, "seconds": float(suite.get("time", 0))}


def main() -> int:
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    suites = {n: run_suite(n, env) for n in SUITES}
    live = "--live" in sys.argv
    suites["live_llm"] = {"status": "NOT RUN", "detail": "needs RUN_LIVE_LLM_TESTS=1, a provider key and network access", "tests": 0, "passed": 0, "failed": 0, "skipped": 0}
    if live:
        env["RUN_LIVE_LLM_TESTS"] = "1"
        suites["live_llm"] = run_suite("live", env) if (ROOT / "tests" / "live").exists() else suites["live_llm"]
    out = {
        "finished_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "python": sys.version.split()[0], "platform": sys.platform, "suites": suites,
        "notes": ["Windows PowerShell commands were never executed on Windows by the build; see README.", "Live provider tests are NOT RUN unless --live is given."],
    }
    (ROOT / "logs" / "test_status.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for n, s in suites.items():
        print(f"{n:12s} {s['status']:8s} passed={s['passed']} failed={s['failed']} skipped={s['skipped']}")
    return 1 if any(s["status"] == "FAILED" for s in suites.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
