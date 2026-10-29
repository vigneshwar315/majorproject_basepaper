"""Interpreter for validated metric specifications. Pure Python, no eval/exec, no I/O."""

from __future__ import annotations

import math
import statistics
from typing import Any

from .metric_specification import MetricSpec, SpecError, parse_spec


class EvalError(RuntimeError):
    """Run-time failure while evaluating a (statically valid) specification."""


def _agg(fn: str, vals: list[float]) -> float:
    if fn == "count":
        return float(len(vals))
    if not vals:
        return 0.0
    if fn == "sum":
        return float(sum(vals))
    if fn == "mean":
        return float(sum(vals) / len(vals))
    if fn == "max":
        return float(max(vals))
    if fn == "min":
        return float(min(vals))
    if fn == "median":
        return float(statistics.median(vals))
    mean = sum(vals) / len(vals)
    if fn == "std":
        return float(math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)))
    if fn == "mad":
        return float(sum(abs(v - mean) for v in vals) / len(vals))
    if fn == "cv":
        if mean == 0:
            return 0.0
        return float(math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) / mean)
    raise EvalError(f"unknown aggregation {fn}")


def _ev(node: dict, view: dict, row: dict | None) -> Any:
    op = node["op"]
    if op == "const":
        return node["value"]
    if op == "field":
        return row[node["name"]]  # type: ignore[index]
    if op == "ctx":
        return view["context"][node["name"]]
    if op == "agg":
        rows = view[node["over"]]
        where = node.get("where")
        vals: list[float] = []
        for r in rows:
            if where is not None and not _ev(where, view, r):
                continue
            if node["fn"] == "count":
                vals.append(1.0)
            else:
                v = _ev(node["value"], view, r)
                vals.append(float(v))
        return _agg(node["fn"], vals)
    if op == "if":
        return _ev(node["then"], view, row) if _ev(node["cond"], view, row) else _ev(node["else"], view, row)
    args = [_ev(a, view, row) for a in node["args"]]
    if op == "add":
        return sum(args)
    if op == "mul":
        out = 1.0
        for a in args:
            out *= a
        return out
    if op == "sub":
        return args[0] - args[1]
    if op == "div":
        if args[1] == 0:
            raise EvalError("division by zero")
        return args[0] / args[1]
    if op == "safe_div":
        return 0.0 if args[1] == 0 else args[0] / args[1]
    if op == "minimum":
        return min(args)
    if op == "maximum":
        return max(args)
    if op == "abs":
        return abs(args[0])
    if op == "neg":
        return -args[0]
    if op == "eq":
        return args[0] == args[1]
    if op == "ne":
        return args[0] != args[1]
    if op == "lt":
        return args[0] < args[1]
    if op == "le":
        return args[0] <= args[1]
    if op == "gt":
        return args[0] > args[1]
    if op == "ge":
        return args[0] >= args[1]
    if op == "and":
        return all(args)
    if op == "or":
        return any(args)
    if op == "not":
        return not args[0]
    raise EvalError(f"unknown op {op}")


def evaluate_spec(spec: MetricSpec | dict | str, view: dict) -> float:
    """Evaluate a specification on a plan view. Raises ``SpecError`` (static) or ``EvalError`` (run time)."""
    if not isinstance(spec, MetricSpec):
        spec = parse_spec(spec)
    try:
        val = _ev(spec.expr, view, None)
    except EvalError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise EvalError(f"{type(exc).__name__}: {exc}") from exc
    val = float(val)
    if not math.isfinite(val):
        raise EvalError("result is not a finite number")
    return val


__all__ = ["EvalError", "SpecError", "evaluate_spec"]
