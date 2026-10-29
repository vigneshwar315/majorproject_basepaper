"""Reference DSL specifications of z1..z14 and keyword evidence used by the *mock* provider.

The DSL specs are our own formalisation of the verbal metric descriptions (Table 2 of the paper). They are checked
against the hand-coded functions in ``reference_metrics.py`` by the unit tests (DSL <-> reference agreement).
"""

from __future__ import annotations

from .metric_specification import MetricSpec, parse_spec


def _f(name):
    return {"op": "field", "name": name}


def _c(value):
    return {"op": "const", "value": value}


def _ctx(name):
    return {"op": "ctx", "name": name}


def _cnt(over, where=None):
    n = {"op": "agg", "fn": "count", "over": over}
    if where is not None:
        n["where"] = where
    return n


def _agg(fn, over, value, where=None):
    n = {"op": "agg", "fn": fn, "over": over, "value": value}
    if where is not None:
        n["where"] = where
    return n


def _op(op, *args):
    return {"op": op, "args": list(args)}


def _and(*a):
    return _op("and", *a)


_mean_today = _agg("mean", "routes", _f("duration"), _f("is_today"))
_mean_tomorrow = _agg("mean", "routes", _f("duration"), _f("is_tomorrow"))

EXPRESSIONS: dict[str, dict] = {
    "z1": _op("safe_div", _cnt("requests", _f("accepted")), _cnt("requests")),
    "z2": _op("safe_div", _cnt("requests", _and(_f("accepted"), _f("arrived_today"))), _cnt("requests", _f("arrived_today"))),
    "z3": _agg("sum", "routes", _f("distance")),
    "z4": _agg("sum", "routes", _f("duration")),
    "z5": _agg("sum", "routes", _f("waiting")),
    "z6": _agg("mad", "routes", _f("duration"), _f("is_today")),
    "z7": _op("mul", _c(100), _op("safe_div", _op("abs", _op("sub", _mean_today, _mean_tomorrow)), _mean_today)),
    "z8": _cnt("visits", _and(_f("in_target_zip"), _f("in_peak"))),
    "z9": _agg("sum", "requests",
               {"op": "if", "cond": _f("accepted"),
                "then": {"op": "if", "cond": _op("le", _f("scheduled_day"), _ctx("current_day")), "then": _c(10), "else": _c(5)},
                "else": _c(0)}),
    "z10": _agg("sum", "requests", {"op": "if", "cond": _f("is_priority"), "then": _c(10), "else": _c(5)}, _f("accepted")),
    "z11": _cnt("requests", _and(_f("accepted"), _f("tight_window"))),
    "z12": _cnt("routes", _f("ends_late")),
    "z13": _cnt("routes", _f("long_next_day")),
    "z14": _cnt("requests", _and(_f("accepted"), _op("gt", _f("customer_num_drivers"), _c(1)))),
}

# Keyword evidence for the deterministic MOCK provider: (phrase, weight). The mock only *simulates* an LLM by
# scoring the metric description against these phrases; results obtained with it are demonstrations of the
# pipeline, never evidence about real LLM behaviour.
KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "z1": [("fraction", 1), ("percentage", 1), ("share", 1), ("proportion", 1), ("accepted", 2), ("accept", 1), ("under the current plan", 1), ("all requests", 1), ("acceptance rate", 2), ("ratio", 1), ("how many requests", 1), ("taken", 0.5), ("overall", 1)],
    "z2": [("fraction", 1), ("percentage", 1), ("proportion", 1), ("accepted", 1), ("today", 3), ("current day", 3), ("same day", 2), ("this day", 2), ("arrived today", 3), ("new requests", 1)],
    "z3": [("total distance", 4), ("distance", 2), ("traveled", 1), ("travelled", 1), ("kilometers", 1), ("km", 1), ("how far", 2), ("miles", 1), ("fleet", 1), ("mileage", 2)],
    "z4": [("total duration", 4), ("duration", 2), ("how actively", 3), ("utilized", 3), ("utilised", 3), ("utilization", 3), ("route time", 2), ("working time", 1), ("busy", 1), ("time on the road", 2), ("hours", 1)],
    "z5": [("waiting", 4), ("wait", 3), ("idle", 3), ("idle time", 3), ("total waiting", 3), ("early arrival", 2), ("standing", 1), ("max(0", 3), ("w_j", 3), ("e_j - a_j", 3)],
    "z6": [("mean absolute deviation", 5), ("mad", 3), ("balanced", 2), ("balance", 2), ("allocation of workload", 2), ("workload among vehicles", 2), ("spread", 1), ("today", 1), ("evenly", 1), ("fair", 1)],
    "z7": [("percentage deviation", 4), ("tomorrow", 3), ("consecutive", 2), ("between today and tomorrow", 3), ("even distribution", 2), ("workdays", 2), ("average route duration", 2), ("difference", 1), ("percent", 1), ("two days", 1)],
    "z8": [("zip", 4), ("postal", 3), ("peak", 3), ("specific times", 2), ("congest", 2), ("visited", 1), ("customers in", 1), ("rush", 2), ("during peak", 2)],
    "z9": [("weighted score", 3), ("score", 2), ("10", 1), ("future date", 3), ("rejected", 1), ("scheduled for today", 3), ("multiple days", 2), ("points", 1), ("0 respectively", 2), ("5", 0.5)],
    "z10": [("priority", 4), ("non-priority", 3), ("high priority", 3), ("weighted score", 1), ("urgent", 2), ("scores of 10 and 5", 3), ("vip", 1), ("importance", 1)],
    "z11": [("tight", 4), ("narrow", 3), ("time window", 2), ("time-window", 2), ("strict", 2)],
    "z12": [("after 18", 3), ("6:00", 3), ("6 pm", 3), ("late", 2), ("return to the depot", 2), ("overtime", 3), ("returning", 2)],
    "z13": [("consecutive days", 4), ("long routes", 3), ("fatigue", 3), ("two days in a row", 3), ("next day", 1)],
    "z14": [("driver", 3), ("customer", 2), ("familiarity", 3), ("continuity", 3), ("more than one", 2), ("different drivers", 3)],
}

METRIC_NAMES = {
    "z1": "fraction_accepted_requests", "z2": "fraction_accepted_today", "z3": "total_distance", "z4": "total_route_duration",
    "z5": "total_waiting_time", "z6": "mad_route_duration_today", "z7": "pct_deviation_avg_duration_today_vs_tomorrow",
    "z8": "customers_in_zip_during_peak", "z9": "weighted_acceptance_score", "z10": "weighted_priority_score",
    "z11": "accepted_tight_window_requests", "z12": "routes_returning_late", "z13": "long_routes_consecutive_days",
    "z14": "customers_with_multiple_drivers",
}


def reference_spec(metric_id: str, description: str = "") -> MetricSpec:
    from .reference_text import DESCRIPTION_OF  # local import to avoid cycle

    return parse_spec({"name": METRIC_NAMES[metric_id], "description": description or DESCRIPTION_OF.get(metric_id, ""), "expr": EXPRESSIONS[metric_id]})


def all_reference_specs() -> dict[str, MetricSpec]:
    return {m: reference_spec(m) for m in EXPRESSIONS}
