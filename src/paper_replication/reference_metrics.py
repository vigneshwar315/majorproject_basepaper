"""Hand-coded reference implementations of the paper's metrics z1..z10 (Table 2, p.149783).

These are independent of the DSL interpreter and serve as ground truth for DMG examples, for the C1/C3
experiments and for the evaluation matrix. z11..z14 are EXTENSION metrics (not in the paper's Table 2) added so
that trade-off prompts 3, 8, 9 and 10 of Table 4 refer to something computable. Empty aggregations return 0.0.
"""

from __future__ import annotations

from typing import Callable

PAPER_METRICS = [f"z{i}" for i in range(1, 11)]
EXTENSION_METRICS = ["z11", "z12", "z13", "z14"]
ALL_METRICS = PAPER_METRICS + EXTENSION_METRICS

DESCRIPTIONS = {
    "z1": "Fraction of accepted requests under the current plan",
    "z2": "Fraction of accepted requests on the current day (i.e., 'today') under the current plan",
    "z3": "Total distance traveled by the fleet",
    "z4": "Total duration of vehicle routes (serving as a measure of how actively the vehicles were utilized)",
    "z5": "Total waiting time of vehicle routes (working time during which the vehicles were idle)",
    "z6": "Mean absolute deviation (MAD) of route durations on the current day (to achieve a balanced allocation of workload among vehicles)",
    "z7": "Percentage deviation of average route durations between 'today' and 'tomorrow' (to ensure an even distribution of workload across consecutive workdays)",
    "z8": "Number of customers in a specific zip code visited at specific times (i.e., during peak times)",
    "z9": "Weighted score based on accepted requests over multiple days (scores of 10, 5, and 0 are assigned to requests scheduled for today, scheduled for a future date, or rejected, respectively)",
    "z10": "Weighted score based on priority vs non-priority accepted requests (scores of 10 and 5 are assigned to scheduled priority and non-priority requests, respectively)",
    "z11": "[extension] Number of accepted requests with a tight time window",
    "z12": "[extension] Number of routes that return to the depot after 18:00",
    "z13": "[extension] Number of long routes whose vehicle also has a long route the next day",
    "z14": "[extension] Number of accepted requests whose customer is served by more than one vehicle",
}
# preferred direction when comparing plans
DIRECTION = {"z1": "max", "z2": "max", "z3": "min", "z4": "min", "z5": "min", "z6": "min", "z7": "min", "z8": "min",
             "z9": "max", "z10": "max", "z11": "min", "z12": "min", "z13": "min", "z14": "min"}
SOURCE = {m: ("paper Table 2" if m in PAPER_METRICS else "extension (not in paper)") for m in ALL_METRICS}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def z1(v):
    reqs = v["requests"]
    return sum(1 for r in reqs if r["accepted"]) / len(reqs) if reqs else 0.0


def z2(v):
    today = [r for r in v["requests"] if r["arrived_today"]]
    return sum(1 for r in today if r["accepted"]) / len(today) if today else 0.0


def z3(v):
    return float(sum(r["distance"] for r in v["routes"]))


def z4(v):
    return float(sum(r["duration"] for r in v["routes"]))


def z5(v):
    return float(sum(r["waiting"] for r in v["routes"]))


def z6(v):
    d = [r["duration"] for r in v["routes"] if r["is_today"]]
    m = _mean(d)
    return _mean([abs(x - m) for x in d])


def z7(v):
    t = _mean([r["duration"] for r in v["routes"] if r["is_today"]])
    n = _mean([r["duration"] for r in v["routes"] if r["is_tomorrow"]])
    return 100.0 * abs(t - n) / t if t else 0.0


def z8(v):
    return float(sum(1 for x in v["visits"] if x["in_target_zip"] and x["in_peak"]))


def z9(v):
    today = v["context"]["current_day"]
    tot = 0.0
    for r in v["requests"]:
        if r["accepted"]:
            tot += 10.0 if r["scheduled_day"] <= today else 5.0
    return tot


def z10(v):
    return float(sum((10 if r["is_priority"] else 5) for r in v["requests"] if r["accepted"]))


def z11(v):
    return float(sum(1 for r in v["requests"] if r["accepted"] and r["tight_window"]))


def z12(v):
    return float(sum(1 for r in v["routes"] if r["ends_late"]))


def z13(v):
    return float(sum(1 for r in v["routes"] if r["long_next_day"]))


def z14(v):
    return float(sum(1 for r in v["requests"] if r["accepted"] and r["customer_num_drivers"] > 1))


REFERENCE: dict[str, Callable[[dict], float]] = {f"z{i}": globals()[f"z{i}"] for i in range(1, 15)}


def compute_metric(metric_id: str, view: dict) -> float:
    return float(REFERENCE[metric_id](view))


def compute_all(view: dict, metrics: list[str] | None = None) -> dict[str, float]:
    return {m: compute_metric(m, view) for m in (metrics or ALL_METRICS)}
