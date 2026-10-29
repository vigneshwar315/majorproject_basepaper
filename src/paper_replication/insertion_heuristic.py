"""Sequential cheapest-insertion heuristic for the multi-period dynamic VRP with pickups and deliveries.

Paper status: the paper names a "simple cheapest-insertion heuristic" and gives no further detail
(PARTIALLY SPECIFIED). This module is therefore an implementation decision (docs/uncertainties_and_deviations.md, U03):

1. Requests are processed in arrival order (dynamic setting); accepted requests are never re-optimised.
2. For a request, every (day, vehicle, pickup position i, delivery position j >= i) is a candidate.
   Raw insertion cost = added distance of inserting the pickup at i and the delivery at j.
3. Candidates are examined in increasing *adjusted* cost (raw cost + policy terms); the first that passes the
   deterministic feasibility check wins (= cheapest feasible insertion). Ties: (cost, day, vehicle index, i, j).
4. If no feasible candidate exists (or a policy rule rejects the request), the request is rejected.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constraints import evaluate_route, finalize_plan
from .problem_instance import DistanceModel, ensure_valid
from .schemas import (
    DELIVERY,
    DEPOT_NODE,
    PICKUP,
    Decision,
    Instance,
    Plan,
    PolicyParams,
    Route,
    Stop,
    Vehicle,
)


@dataclass
class _RouteCands:
    day: int
    vidx: int
    vehicle: Vehicle
    stops: list[Stop]
    width: int  # L + 1
    order: np.ndarray
    adjusted: np.ndarray
    raw: np.ndarray


def insertion_cost_matrix(dm: DistanceModel, stops: list[Stop], pickup_node: str, delivery_node: str) -> np.ndarray:
    """Raw added-distance matrix M[i, j] for inserting pickup before original position i and delivery before
    original position j (j >= i). Entries with j < i are +inf. Position L means 'after the last stop'."""
    D = dm.matrix
    depot = dm.index[DEPOT_NODE]
    seq = [depot] + [dm.index[s.node] for s in stops] + [depot]
    a = np.array(seq[:-1])
    b = np.array(seq[1:])
    p = dm.index[pickup_node]
    q = dm.index[delivery_node]
    base = D[a, b]
    cost_p = D[a, p] + D[p, b] - base
    cost_d = D[a, q] + D[q, b] - base
    adj = D[a, p] + D[p, q] + D[q, b] - base  # pickup and delivery adjacent
    m = cost_p[:, None] + cost_d[None, :]
    n = len(a)
    iu = np.triu(np.ones((n, n), dtype=bool), k=1)
    m = np.where(iu, m, np.inf)
    m[np.arange(n), np.arange(n)] = adj
    return m


def build_plan(inst: Instance, policy: Optional[PolicyParams] = None, plan_id: str = "plan_1", dm: Optional[DistanceModel] = None) -> Plan:
    """Run the heuristic over all received requests and return a validated plan."""
    ensure_valid(inst)
    policy = policy or PolicyParams()
    dm = dm or DistanceModel(inst)
    reqs = inst.request_map()
    rng = np.random.default_rng(policy.noise_seed)
    routes: dict[tuple[int, str], list[Stop]] = {}
    durations: dict[tuple[int, str], float] = {}
    decisions: dict[str, Decision] = {}
    last_day = inst.horizon_days - 1
    for r in inst.received_requests():
        mw = r.min_tw_width()
        if policy.tight_tw_threshold is not None and mw is not None and mw <= policy.tight_tw_threshold:
            decisions[r.id] = Decision(r.id, "rejected", reason="policy: tight time window")
            continue
        pnode, dnode = Instance.node_id(r.id, PICKUP), Instance.node_id(r.id, DELIVERY)
        cands: list[_RouteCands] = []
        for day in range(r.arrival_day, min(r.arrival_day + inst.max_lead_days, last_day) + 1):
            for vidx, v in enumerate(inst.vehicles):
                if r.demand > v.capacity:
                    continue
                stops = routes.get((day, v.id), [])
                raw = insertion_cost_matrix(dm, stops, pnode, dnode).ravel()
                adjusted = raw + policy.day_penalty * (day - r.arrival_day) + policy.balance_weight * durations.get((day, v.id), 0.0)
                if policy.noise > 0:
                    adjusted = adjusted + rng.uniform(0.0, policy.noise, size=adjusted.shape)
                keys = np.round(np.where(np.isfinite(adjusted), adjusted, np.inf), 9)
                order = np.lexsort((np.arange(len(keys)), keys))
                n_valid = int(np.isfinite(keys).sum())
                cands.append(_RouteCands(day, vidx, v, stops, len(stops) + 1, order[:n_valid], keys, raw))
        heap: list[tuple[float, int, int, int]] = []
        for ci, c in enumerate(cands):
            if len(c.order):
                heapq.heappush(heap, (float(c.adjusted[c.order[0]]), c.day, c.vidx, ci * 10_000_000))
        chosen = None
        reason = "no feasible insertion"
        while heap:
            _, _, _, code = heapq.heappop(heap)
            ci, k = divmod(code, 10_000_000)
            c = cands[ci]
            f = int(c.order[k])
            if k + 1 < len(c.order):
                nxt = int(c.order[k + 1])
                heapq.heappush(heap, (float(c.adjusted[nxt]), c.day, c.vidx, ci * 10_000_000 + k + 1))
            if policy.max_detour is not None and c.stops and c.raw[f] > policy.max_detour + 1e-9:
                reason = "policy: detour above limit"
                continue
            i, j = divmod(f, c.width)
            new_stops = c.stops[:i] + [Stop(r.id, PICKUP)] + c.stops[i:j] + [Stop(r.id, DELIVERY)] + c.stops[j:]
            res = evaluate_route(inst, dm, reqs, c.vehicle, new_stops, policy=policy, stop_at_first=True)
            if res.feasible:
                full = evaluate_route(inst, dm, reqs, c.vehicle, new_stops, policy=policy)
                chosen = (c, new_stops, full.duration)
                break
        if chosen is None:
            decisions[r.id] = Decision(r.id, "rejected", reason=reason)
        else:
            c, new_stops, dur = chosen
            routes[(c.day, c.vehicle.id)] = new_stops
            durations[(c.day, c.vehicle.id)] = dur
            decisions[r.id] = Decision(r.id, "accepted", day=c.day, vehicle_id=c.vehicle.id)
    vorder = {v.id: i for i, v in enumerate(inst.vehicles)}
    route_list = [Route(d, vid, stops) for (d, vid), stops in sorted(routes.items(), key=lambda kv: (kv[0][0], vorder[kv[0][1]])) if stops]
    plan = Plan(plan_id, policy, route_list, decisions)
    return finalize_plan(inst, plan, dm)
