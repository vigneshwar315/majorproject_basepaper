"""Generation of m alternative plans (paper p.149783: "the VRP heuristic returns m alternative solutions").

How the paper obtains its m alternatives is NOT SPECIFIED. Here: the first plan is the plain cheapest-insertion
baseline; the others come from the same heuristic under sampled policy parameters (day preference, detour cap,
tight-window rejection, shift limit, reserved capacity, peak-hour ZIP avoidance, balance weight, cost noise).
Duplicate plans are discarded so every candidate is distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .insertion_heuristic import build_plan
from .problem_instance import DistanceModel
from .schemas import Instance, Plan, PolicyParams


@dataclass
class CandidateSet:
    plans: list[Plan]
    requested_m: int
    seed: int
    tries: int
    notes: list[str] = field(default_factory=list)

    @property
    def produced_m(self) -> int:
        return len(self.plans)


def sample_policy(rng: np.random.Generator, zips: list[str], index: int, noise_seed: int) -> PolicyParams:
    def pick(options):
        return options[int(rng.integers(0, len(options)))]

    return PolicyParams(
        label=f"sampled_{index}",
        day_penalty=float(pick([0, 0, 10, 25, 50, 100, 250])),
        max_detour=pick([None, None, None, 90.0, 120.0, 180.0]),
        tight_tw_threshold=pick([None, None, None, None, 30.0, 45.0]),
        shift_limit=pick([None, None, None, 420.0, 450.0, 465.0]),
        reserve_fraction=float(pick([0.0, 0.0, 0.0, 0.17, 0.34])),
        avoid_zip=pick([None, None] + list(zips)) if zips else None,
        balance_weight=float(pick([0.0, 0.0, 0.1, 0.3, 0.6, 1.0])),
        noise=float(pick([0.0, 5.0, 15.0, 30.0, 60.0])),
        noise_seed=noise_seed,
    )


def generate_candidates(inst: Instance, m: int, seed: int = 0, max_tries_factor: int = 8, dm: Optional[DistanceModel] = None) -> CandidateSet:
    """Return up to ``m`` distinct, validated candidate plans named pi_1..pi_m."""
    if m < 1:
        raise ValueError("m must be >= 1")
    dm = dm or DistanceModel(inst)
    rng = np.random.default_rng(seed)
    zips = inst.zips()
    plans: list[Plan] = []
    seen: set[str] = set()
    tries = 0
    base = build_plan(inst, PolicyParams(label="baseline"), "tmp", dm)
    seen.add(base.fingerprint())
    base.plan_id = "pi_1"
    plans.append(base)
    tries += 1
    while len(plans) < m and tries < max_tries_factor * m:
        pol = sample_policy(rng, zips, tries, noise_seed=int(rng.integers(0, 2**31 - 1)))
        plan = build_plan(inst, pol, "tmp", dm)
        tries += 1
        fp = plan.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        plan.plan_id = f"pi_{len(plans) + 1}"
        plans.append(plan)
    notes = []
    if len(plans) < m:
        notes.append(f"only {len(plans)} distinct plans found for requested m={m} after {tries} heuristic runs")
    return CandidateSet(plans, m, seed, tries, notes)
