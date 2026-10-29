"""Constrained metric DSL (replaces the paper's LLM-generated executable code; see deviation U01).

A metric is a JSON document ``{"name", "description", "expr"}`` whose ``expr`` is an expression tree over the
*plan view* (``plan_view.build_plan_view``): entity tables ``routes`` / ``requests`` / ``visits`` and context
constants. Only whitelisted operators exist, so there is no way to execute arbitrary code or touch the file
system / network. Every specification is statically validated before it is ever evaluated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

MAX_DEPTH = 12
MAX_NODES = 120

FIELDS: dict[str, dict[str, str]] = {
    "routes": {
        "vehicle_id": "str", "day": "num", "distance": "num", "duration": "num", "waiting": "num",
        "num_stops": "num", "num_requests": "num", "start_time": "num", "end_time": "num", "max_load": "num",
        "is_today": "bool", "is_tomorrow": "bool", "ends_late": "bool", "is_long": "bool", "long_next_day": "bool",
    },
    "requests": {
        "id": "str", "customer": "str", "priority": "num", "is_priority": "bool", "accepted": "bool",
        "scheduled_day": "num", "arrival_day": "num", "arrived_today": "bool", "scheduled_today": "bool",
        "scheduled_future": "bool", "has_time_window": "bool", "tight_window": "bool", "vehicle_id": "str",
        "customer_num_drivers": "num",
    },
    "visits": {
        "request_id": "str", "kind": "str", "zip": "str", "day": "num", "vehicle_id": "str", "arrival": "num",
        "service_start": "num", "wait": "num", "is_today": "bool", "in_peak": "bool", "in_target_zip": "bool",
    },
}
CONTEXT: dict[str, str] = {
    "current_day": "num", "tomorrow": "num", "num_vehicles": "num", "num_requests_received": "num",
    "peak_start": "num", "peak_end": "num", "target_zip": "str", "late_return_time": "num",
    "long_route_minutes": "num", "tight_window_width": "num",
}
AGG_FUNCS = ("sum", "count", "mean", "max", "min", "median", "std", "mad", "cv")
ARITH_BINARY = ("add", "sub", "mul", "div", "safe_div", "minimum", "maximum")
ARITH_UNARY = ("abs", "neg")
COMPARE = ("eq", "ne", "lt", "le", "gt", "ge")
LOGIC = ("and", "or", "not")

SPEC_JSON_SCHEMA = {
    "type": "object",
    "properties": {"spec_json": {"type": "string", "description": "JSON text of the metric specification (see DSL reference)"}},
    "required": ["spec_json"],
}


class SpecError(ValueError):
    """A metric specification is malformed or not allowed. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(errors))


@dataclass(frozen=True)
class MetricSpec:
    name: str
    description: str
    expr: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "expr": self.expr}

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def fingerprint(self) -> str:
        """Canonical form of the expression only (name/description ignored)."""
        return json.dumps(self.expr, sort_keys=True, separators=(",", ":"))


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_spec(source: Any) -> MetricSpec:
    """Parse + statically validate a specification (JSON text or dict). Raises ``SpecError``."""
    if isinstance(source, MetricSpec):
        obj = source.to_dict()
    elif isinstance(source, dict):
        obj = source
    elif isinstance(source, str):
        text = source.strip()
        m = _FENCE.match(text)
        if m:
            text = m.group(1)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecError([f"specification is not valid JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"]) from exc
    else:
        raise SpecError(["specification must be a JSON object"])
    if not isinstance(obj, dict):
        raise SpecError(["specification must be a JSON object"])
    errs = validate_spec_dict(obj)
    if errs:
        raise SpecError(errs)
    return MetricSpec(str(obj.get("name", "metric")), str(obj.get("description", "")), obj["expr"])


def validate_spec_dict(obj: dict) -> list[str]:
    errs: list[str] = []
    extra = set(obj) - {"name", "description", "expr"}
    if extra:
        errs.append(f"unknown top-level keys: {sorted(extra)}")
    if "expr" not in obj:
        errs.append("missing 'expr'")
        return errs
    counter = [0]
    t = _check(obj["expr"], None, 1, counter, errs, "expr")
    if not errs and t != "num":
        errs.append(f"the metric expression must be numeric, got {t}")
    return errs


def _check(node: Any, scope: Optional[str], depth: int, counter: list[int], errs: list[str], path: str) -> str:
    """Return the static type of ``node`` ('num'|'bool'|'str'|'err'), appending problems to ``errs``."""
    counter[0] += 1
    if counter[0] > MAX_NODES:
        if counter[0] == MAX_NODES + 1:
            errs.append(f"expression has more than {MAX_NODES} nodes")
        return "err"
    if depth > MAX_DEPTH:
        errs.append(f"{path}: expression deeper than {MAX_DEPTH} levels")
        return "err"
    if not isinstance(node, dict) or "op" not in node:
        errs.append(f"{path}: node must be an object with an 'op'")
        return "err"
    op = node["op"]
    keys = set(node) - {"op"}

    def only(allowed: set[str]) -> bool:
        bad = keys - allowed
        if bad:
            errs.append(f"{path}: unexpected keys {sorted(bad)} for op {op!r}")
            return False
        return True

    if op == "const":
        only({"value"})
        v = node.get("value")
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, (int, float)):
            if v != v or v in (float("inf"), float("-inf")):
                errs.append(f"{path}: constant must be finite")
                return "err"
            return "num"
        if isinstance(v, str):
            if len(v) > 64:
                errs.append(f"{path}: string constant too long")
            return "str"
        errs.append(f"{path}: const needs a number, boolean or string 'value'")
        return "err"
    if op == "field":
        only({"name"})
        if scope is None:
            errs.append(f"{path}: 'field' is only allowed inside an aggregation")
            return "err"
        name = node.get("name")
        if name not in FIELDS[scope]:
            errs.append(f"{path}: unknown field {name!r} for entity {scope!r}; allowed: {sorted(FIELDS[scope])}")
            return "err"
        return FIELDS[scope][name]
    if op == "ctx":
        only({"name"})
        name = node.get("name")
        if name not in CONTEXT:
            errs.append(f"{path}: unknown context constant {name!r}; allowed: {sorted(CONTEXT)}")
            return "err"
        return CONTEXT[name]
    if op == "agg":
        only({"fn", "over", "where", "value"})
        if scope is not None:
            errs.append(f"{path}: aggregations cannot be nested inside aggregations")
            return "err"
        fn, over = node.get("fn"), node.get("over")
        if fn not in AGG_FUNCS:
            errs.append(f"{path}: unknown aggregation {fn!r}; allowed: {list(AGG_FUNCS)}")
        if over not in FIELDS:
            errs.append(f"{path}: unknown entity {over!r}; allowed: {list(FIELDS)}")
            return "err"
        if node.get("where") is not None:
            t = _check(node["where"], over, depth + 1, counter, errs, path + ".where")
            if t not in ("bool", "err"):
                errs.append(f"{path}.where: must be boolean, got {t}")
        if fn == "count":
            if node.get("value") is not None:
                errs.append(f"{path}: 'count' takes no 'value'")
        elif fn in AGG_FUNCS:
            if node.get("value") is None:
                errs.append(f"{path}: aggregation {fn!r} requires 'value'")
            else:
                t = _check(node["value"], over, depth + 1, counter, errs, path + ".value")
                if t not in ("num", "err"):
                    errs.append(f"{path}.value: must be numeric, got {t}")
        return "num"
    if op in ARITH_BINARY or op in ARITH_UNARY or op in COMPARE or op in LOGIC:
        only({"args"})
        args = node.get("args")
        if not isinstance(args, list):
            errs.append(f"{path}: 'args' must be a list")
            return "err"
        want = 1 if (op in ARITH_UNARY or op == "not") else 2
        multi = op in ("add", "mul", "and", "or")
        if (multi and len(args) < 2) or (not multi and len(args) != want):
            errs.append(f"{path}: op {op!r} takes {'2 or more' if multi else want} argument(s), got {len(args)}")
            return "err"
        types = [_check(a, scope, depth + 1, counter, errs, f"{path}.args[{i}]") for i, a in enumerate(args)]
        if "err" in types:
            return "err"
        if op in ARITH_BINARY or op in ARITH_UNARY:
            if any(t != "num" for t in types):
                errs.append(f"{path}: op {op!r} needs numeric arguments, got {types}")
                return "err"
            return "num"
        if op in ("lt", "le", "gt", "ge"):
            if any(t != "num" for t in types):
                errs.append(f"{path}: op {op!r} needs numeric arguments, got {types}")
                return "err"
            return "bool"
        if op in ("eq", "ne"):
            if types[0] != types[1]:
                errs.append(f"{path}: op {op!r} compares values of different types {types}")
                return "err"
            return "bool"
        if any(t != "bool" for t in types):
            errs.append(f"{path}: op {op!r} needs boolean arguments, got {types}")
            return "err"
        return "bool"
    if op == "if":
        only({"cond", "then", "else"})
        for k in ("cond", "then", "else"):
            if k not in node:
                errs.append(f"{path}: 'if' requires '{k}'")
                return "err"
        tc = _check(node["cond"], scope, depth + 1, counter, errs, path + ".cond")
        tt = _check(node["then"], scope, depth + 1, counter, errs, path + ".then")
        te = _check(node["else"], scope, depth + 1, counter, errs, path + ".else")
        if "err" in (tc, tt, te):
            return "err"
        if tc != "bool":
            errs.append(f"{path}.cond: must be boolean, got {tc}")
        if tt != te:
            errs.append(f"{path}: 'then' and 'else' must have the same type ({tt} vs {te})")
            return "err"
        return tt
    errs.append(f"{path}: unknown op {op!r}")
    return "err"


def dsl_reference() -> str:
    """Human/LLM-readable description of the DSL (embedded in the generation prompt)."""
    lines = [
        "A metric specification is a JSON object: {\"name\": str, \"description\": str, \"expr\": <node>}.",
        "Every node is an object with an \"op\". Allowed nodes:",
        '  {"op":"const","value":<number|bool|string>}',
        '  {"op":"field","name":<field>}            (only inside agg "where"/"value"; refers to the current row)',
        '  {"op":"ctx","name":<context constant>}',
        '  {"op":"agg","fn":<fn>,"over":<entity>,"where":<bool node, optional>,"value":<numeric node, not for count>}',
        f"     fn in {list(AGG_FUNCS)} (empty selection gives 0.0; mad = mean absolute deviation around the mean; cv = std/mean)",
        f'  {{"op":<one of {list(ARITH_BINARY)}>,"args":[a,b]}}  (add/mul accept 2 or more args; safe_div returns 0 if the divisor is 0; div fails)',
        f'  {{"op":<one of {list(ARITH_UNARY)}>,"args":[a]}}',
        f'  {{"op":<one of {list(COMPARE)}>,"args":[a,b]}}   (lt/le/gt/ge numeric; eq/ne same type)',
        '  {"op":"and"|"or","args":[...]}   {"op":"not","args":[a]}',
        '  {"op":"if","cond":<bool>,"then":<node>,"else":<node>}',
        "The top-level expression must be numeric. Aggregations cannot be nested inside aggregations.",
        "Entities and fields:",
    ]
    for ent, flds in FIELDS.items():
        lines.append(f"  {ent}: " + ", ".join(f"{k}:{v}" for k, v in flds.items()))
    lines.append("Context constants: " + ", ".join(f"{k}:{v}" for k, v in CONTEXT.items()))
    lines.append("Plan semantics: 'today' = context current_day; a request is accepted when requests.accepted is true;"
                 " scheduled_day is -1 for rejected requests; times are minutes since midnight.")
    return "\n".join(lines)
