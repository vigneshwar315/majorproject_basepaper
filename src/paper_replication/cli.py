"""Command-line entry point:  python -m paper_replication.cli <command>."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .config import ConfigError, Settings, settings_from_env
from .experiments.runner import EXPERIMENTS, run_all, run_experiment
from .llm import get_provider
from .llm.base import LLMError
from .pipeline import MetricRequest, run_pipeline
from .problem_instance import generate_instance
from .redact import redact

DEMO_PREFERENCE = ("Maximize the number of accepted requests and prioritize servicing new requests as soon as possible (e.g., today rather than "
                   "tomorrow). However, maintaining route balance is essential—avoid overloading some vehicles while underutilizing others.")
DEMO_METRIC = "Introduce a new metric: the total waiting time of the vehicle routes, i.e., the working time during which the vehicles are idle."


def _settings(args) -> Settings:
    s = settings_from_env()
    over = {}
    if getattr(args, "provider", None):
        over["provider"] = args.provider
    if getattr(args, "model", None):
        over["model"] = args.model
    return dataclasses.replace(s, **over) if over else s


def _provider(args):
    s = _settings(args)
    p = get_provider(s)
    if not s.is_mock and getattr(args, "verify", True):
        p.verify_model()
    return p


def cmd_config(args) -> int:
    print(json.dumps(_settings(args).public_summary(), indent=2))
    return 0


def cmd_list_models(args) -> int:
    s = _settings(args)
    if s.is_mock:
        print("mock-deterministic")
        return 0
    if not s.has_credentials():
        raise ConfigError("API key missing for provider %s" % s.provider)
    p = get_provider(dataclasses.replace(s, model=s.model or "placeholder"))
    for name in sorted(p.list_models()):
        print(name)
    return 0


def cmd_demo(args) -> int:
    p = _provider(args)
    inst = generate_instance(seed=args.seed, n_requests=args.requests)
    res = run_pipeline(inst, args.preference or DEMO_PREFERENCE, p, metric_requests=[MetricRequest(DEMO_METRIC, oracle="z5")], m=args.m, seed=args.seed)
    print(f"run_id      : {res.run_id}")
    print(f"category    : {res.category}" + ("   [MOCK PROVIDER - not an LLM]" if p.is_mock else ""))
    print(f"candidates  : {res.candidate_set.produced_m} (feasible: {sum(1 for x in res.plans if x.feasible)})")
    for k, d in res.dmg_results.items():
        print(f"DMG {k}      : {d.status} after {d.attempts_used} attempt(s)")
    if res.mcdm:
        print(f"MCDM        : {res.mcdm.status}, selected {res.mcdm.selected_plan_id}")
        print(f"explanation : {res.mcdm.justification}")
    for w in res.warnings:
        print("warning     :", w)
    print(f"artefacts   : {res.out_dir}")
    return 0 if res.mcdm and res.mcdm.success else 2


def cmd_run_experiment(args) -> int:
    p = _provider(args)
    prog = (lambda i, n, msg: print(f"\r[{i}/{n}] {msg}      ", end="", flush=True)) if sys.stdout.isatty() else None
    res = run_experiment(args.exp_id.upper(), p, seed=args.seed, max_attempts=args.max_attempts, progress=prog)
    print()
    print(f"{res.exp_id} finished: category={res.category}, artefacts in {res.out_dir}")
    print(json.dumps(res.summary, indent=2, default=str)[:4000])
    return 0


def cmd_run_all(args) -> int:
    p = _provider(args)
    for r in run_all(p, ids=[i.upper() for i in args.only] if args.only else None, seed=args.seed, max_attempts=args.max_attempts):
        print(f"{r.exp_id}: category={r.category} -> {r.out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="paper_replication", description="Reproduction of Ghiani et al. (IEEE Access 2025)")
    ap.add_argument("--provider", choices=["mock", "gemini", "openai"], help="override LLM_PROVIDER")
    ap.add_argument("--model", help="override LLM_MODEL")
    ap.add_argument("--no-verify", dest="verify", action="store_false", help="skip checking the model id against the provider")
    # the same three options are accepted after the sub-command too (`run-all --provider mock`)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", choices=["mock", "gemini", "openai"], default=argparse.SUPPRESS, help="override LLM_PROVIDER")
    common.add_argument("--model", default=argparse.SUPPRESS, help="override LLM_MODEL")
    common.add_argument("--no-verify", dest="verify", action="store_false", default=argparse.SUPPRESS, help="skip checking the model id")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("config", parents=[common], help="show provider/model status (no secrets)").set_defaults(fn=cmd_config)
    sub.add_parser("list-models", parents=[common], help="list model ids offered by the configured provider").set_defaults(fn=cmd_list_models)
    d = sub.add_parser("demo", parents=[common], help="one end-to-end run: routing -> DMG -> MCDM")
    d.add_argument("--seed", type=int, default=20251028)
    d.add_argument("--requests", type=int, default=60)
    d.add_argument("--m", type=int, default=10)
    d.add_argument("--preference")
    d.set_defaults(fn=cmd_demo)
    e = sub.add_parser("run-experiment", parents=[common], help="run one experiment (C1..C6)")
    e.add_argument("exp_id", choices=sorted(EXPERIMENTS) + [k.lower() for k in EXPERIMENTS])
    e.add_argument("--seed", type=int, default=20251028)
    e.add_argument("--max-attempts", type=int, default=5)
    e.set_defaults(fn=cmd_run_experiment)
    a = sub.add_parser("run-all", parents=[common], help="run the complete reproduction protocol (C1..C6)")
    a.add_argument("--only", nargs="*", help="subset of experiment ids")
    a.add_argument("--seed", type=int, default=20251028)
    a.add_argument("--max-attempts", type=int, default=5)
    a.set_defaults(fn=cmd_run_all)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (ConfigError, LLMError) as exc:
        print("error:", redact(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
