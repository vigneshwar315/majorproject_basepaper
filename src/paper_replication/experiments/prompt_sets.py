"""Prompt sets for the experiments.

Paper: seed metric prompts = Table 2; seed trade-off prompts = Table 4 (9 prompts, numbered 1,2,3,5-10). The paper then
used an LLM to produce 30 paraphrases per prompt type and LLM-made degradations. Those LLM-made sets are NOT AVAILABLE.
Offline we therefore use HAND-WRITTEN paraphrases and RULE-BASED degradations (labelled ``source`` in every row). When
a real provider is configured, users may substitute LLM-generated sets; this is not done automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..reference_metrics import DESCRIPTIONS, PAPER_METRICS

# ---------------------------------------------------------------------------------------------------------------
# Metric prompts (C1/C3)
# ---------------------------------------------------------------------------------------------------------------
SEED_METRIC_TEXT = {m: DESCRIPTIONS[m] for m in PAPER_METRICS}  # Table 2 descriptions

T_PARAPHRASES: dict[str, list[str]] = {
    "z1": [
        "Introduce a new metric: the fraction of all received requests that the current plan accepts.",
        "Add a metric that reports what share of the requests received so far has been accepted in the current plan.",
        "I would like a new measure of the acceptance rate of the plan: accepted requests divided by all requests.",
    ],
    "z2": [
        "Introduce a new metric: the fraction of the requests received today that are accepted in the current plan.",
        "Add a metric giving the percentage of today's new requests that the plan accepts, where today is the current day.",
        "Create a measure of the proportion of accepted requests among those that arrived on the current day.",
    ],
    "z3": [
        "Introduce a new metric: the total distance traveled by all vehicles of the fleet in the plan.",
        "Add a metric that sums the distance covered by every vehicle route in the plan.",
        "I need a new measure: the overall mileage driven by the fleet under the current plan.",
    ],
    "z4": [
        "Introduce a new metric: the total duration of all the vehicle routes, to see how actively the vehicles are utilized.",
        "Add a metric that adds up the durations of the vehicle routes in the plan; it measures how busy the vehicles are.",
        "Create a measure of fleet utilization: the sum of the time that each vehicle spends on its route.",
    ],
    "z5": [
        "Introduce a new metric: the total waiting time of the vehicle routes, i.e., the working time during which the vehicles are idle.",
        "Add a metric summing, over all routes, the time vehicles spend idle waiting for a customer's time window to open.",
        "Create a measure of the overall idle time of the fleet while waiting at customers.",
    ],
    "z6": [
        "Introduce a new metric to evaluate the balance of today's fleet plan: the mean absolute deviation of the durations of the routes scheduled for today.",
        "Add a metric measuring how evenly the workload is spread among vehicles today: the average absolute deviation of today's route durations from their mean.",
        "Create a balance indicator for the current day based on the mean absolute deviation of the route durations.",
    ],
    "z7": [
        "Introduce a new metric: the percentage deviation between the average route duration of today and that of tomorrow, to keep the workload even across consecutive workdays.",
        "Add a metric that compares the average route duration today with tomorrow's and reports the difference as a percentage of today's average.",
        "Create a measure of workload evenness over two consecutive days: the percent difference of average route durations between today and tomorrow.",
    ],
    "z8": [
        "Introduce a new metric: the number of customers in a specific zip code visited during peak times.",
        "Add a metric counting how many customer visits fall in a given zip code during the peak-hour window.",
        "Create a measure that counts the visits made in the chosen postal code while it is rush hour.",
    ],
    "z9": [
        "Introduce a new metric: a weighted score over the requests, giving 10 points to requests accepted and scheduled for today, 5 to those scheduled for a future date, and 0 to rejected ones.",
        "Add a score that sums 10 for each request scheduled today, 5 for each request scheduled on a later day, and nothing for rejected requests.",
        "Create a measure summing points per request over multiple days: 10 if served today, 5 if served in the future, 0 if rejected.",
    ],
    "z10": [
        "Introduce a new metric: a weighted score of accepted requests that gives 10 points to each priority request and 5 points to each non-priority request.",
        "Add a score summing 10 for every accepted high-priority request and 5 for every accepted normal request.",
        "Create a measure rewarding the acceptance of priority requests: 10 per scheduled priority request, 5 per scheduled non-priority one.",
    ],
}

FORMULAE: dict[str, str] = {
    "z1": r"Let $R$ be the set of received requests and $A \subseteq R$ the accepted ones. The metric is $z_1 = |A| / |R|$.",
    "z2": r"Let $R_0$ be the set of requests received today and $A_0 \subseteq R_0$ those accepted. The metric is $z_2 = |A_0| / |R_0|$.",
    "z3": r"Let $d_k$ be the distance of route $k$ in the plan. The metric is $z_3 = \sum_k d_k$.",
    "z4": r"Let $D_k$ be the duration of route $k$, from its departure from the depot to its return. The metric is $z_4 = \sum_k D_k$.",
    "z5": r"For the stops $j$ of a route, with arrival time $a_j = d_i + t_{ij}$, earliest service time $e_j$ and waiting time $w_j = \max(0, e_j - a_j)$, the metric is $z_5 = \sum_j w_j$ over all stops of all routes.",
    "z6": r"Let $D_1,\dots,D_k$ be the durations of today's routes and $\bar{D}$ their mean. The metric is $z_6 = \frac{1}{k}\sum_i |D_i - \bar{D}|$.",
    "z7": r"Let $\bar{D}_0$ and $\bar{D}_1$ be the average route durations of today and tomorrow. The metric is $z_7 = 100\cdot|\bar{D}_0 - \bar{D}_1| / \bar{D}_0$.",
    "z8": r"Let $V$ be the set of visits to the target zip code whose service start lies in the peak window. The metric is $z_8 = |V|$.",
    "z9": r"$z_9 = \sum_{r} s_r$ with $s_r = 10$ if request $r$ is scheduled today, $s_r = 5$ if it is scheduled on a later day and $s_r = 0$ if it is rejected.",
    "z10": r"$z_{10} = \sum_{r \in A} s_r$ over the accepted requests $A$, with $s_r = 10$ for priority requests and $s_r = 5$ for non-priority requests.",
}

PSEUDOCODE: dict[str, str] = {
    "z1": r"\begin{enumerate} \item $acc \leftarrow 0$ \item for each received request: if accepted then $acc \leftarrow acc + 1$ \item return $acc / (\text{number of received requests})$ \end{enumerate}",
    "z2": r"\begin{enumerate} \item $n \leftarrow 0$, $acc \leftarrow 0$ \item for each request received today: $n \leftarrow n + 1$; if accepted then $acc \leftarrow acc + 1$ \item return $acc / n$ \end{enumerate}",
    "z3": r"\begin{enumerate} \item $total \leftarrow 0$ \item for each route: $total \leftarrow total + \text{distance of the route}$ \item return $total$ \end{enumerate}",
    "z4": r"\begin{enumerate} \item $total \leftarrow 0$ \item for each route: $total \leftarrow total + (\text{return time} - \text{departure time})$ \item return $total$ \end{enumerate}",
    "z5": r"\begin{enumerate} \item $total \leftarrow 0$ \item for each route and each stop $j$: $a_j \leftarrow d_i + t_{ij}$; $w_j \leftarrow \max(0, e_j - a_j)$; $total \leftarrow total + w_j$ \item return $total$ \end{enumerate}",
    "z6": r"\begin{enumerate} \item collect the durations $D_i$ of today's routes \item $\bar{D} \leftarrow$ mean of the $D_i$ \item return the mean of $|D_i - \bar{D}|$ \end{enumerate}",
    "z7": r"\begin{enumerate} \item $m_0 \leftarrow$ mean duration of today's routes \item $m_1 \leftarrow$ mean duration of tomorrow's routes \item return $100 \cdot |m_0 - m_1| / m_0$ (0 if $m_0 = 0$) \end{enumerate}",
    "z8": r"\begin{enumerate} \item $c \leftarrow 0$ \item for each visit: if its zip code is the target and its service start is within the peak window then $c \leftarrow c + 1$ \item return $c$ \end{enumerate}",
    "z9": r"\begin{enumerate} \item $score \leftarrow 0$ \item for each request: if accepted and scheduled today add 10, else if accepted add 5, else add 0 \item return $score$ \end{enumerate}",
    "z10": r"\begin{enumerate} \item $score \leftarrow 0$ \item for each accepted request: if priority add 10 else add 5 \item return $score$ \end{enumerate}",
}


@dataclass(frozen=True)
class MetricPrompt:
    prompt_id: str
    metric_id: str
    ptype: str  # T | TF | TFP
    text: str
    source: str  # "table2_seed" | "handwritten_paraphrase" | "rule_degraded:<level>"
    degradation: str = "none"


def metric_prompt_set(ptype: str) -> list[MetricPrompt]:
    """30 prompts of type T/TF/TFP: 10 seed metrics x 3 hand-written paraphrases (paper: 30 per type)."""
    if ptype not in ("T", "TF", "TFP"):
        raise ValueError("ptype must be T, TF or TFP")
    out = []
    for m in PAPER_METRICS:
        for i, base in enumerate(T_PARAPHRASES[m], start=1):
            text = base
            if ptype in ("TF", "TFP"):
                text += "\n" + FORMULAE[m]
            if ptype == "TFP":
                text += "\nPseudocode:\n" + PSEUDOCODE[m]
            out.append(MetricPrompt(f"{ptype}-{m}-{i}", m, ptype, text, "handwritten_paraphrase"))
    return out


# ---- rule-based degradation of the textual part (paper: LLM-made degradations, NOT AVAILABLE) -------------------
_INFORMAL = [
    (r"^Introduce a new metric( to evaluate)?:?", "hey, can u add a metric"),
    (r"^Add a (metric|score)( that)?", r"pls add a \1 that"),
    (r"^Create a (measure|balance indicator)( of| for)?", r"make me a \1 for"),
    (r"^I (would like|need) a new measure( of)?:?", "gimme a number for"),
    (r"\bthe number of\b", "how many"),
    (r"\bpercentage\b", "%"),
    (r"\btotal\b", "total"),
    (r"\bthe fraction of\b", "how much of"),
    (r"\bto see how\b", "so we know how"),
    (r"\.$", " thx"),
]
_VAGUE = [
    (r"\baccepted\b", "handled"), (r"\baccepts\b", "handles"), (r"\bdistance\b", "length"), (r"\btraveled\b", "done"),
    (r"\bwaiting\b", "hanging around"), (r"\bidle\b", "unproductive"), (r"\bpriority\b", "important"), (r"\bhigh-priority\b", "important"),
    (r"\bzip code\b", "area"), (r"\bpostal code\b", "area"), (r"\bdeviation\b", "difference"), (r"\bduration(s)?\b", "time"),
    (r"\bmean absolute\b", "typical"), (r"\bpeak(-hour| times)?\b", "busy"), (r"\bvehicle routes\b", "trips"), (r"\broutes\b", "trips"),
    (r"\bscheduled\b", "planned"), (r"\bfraction\b", "part"),
]
_AMBIG = [
    (r"\b(?:the )?(?:today|current day)(?:'s)?\b", "the relevant day"), (r"\b(?:the )?tomorrow(?:'s)?\b", "the other day"),
    (r"\bcurrent plan\b", "the plan"), (r"\b(10|5|0) points?\b", "some points"), (r"\b10\b", "a number"), (r"\bfuture date\b", "another date"),
    (r"\bpriority\b", "special"), (r"\bnon-special\b", "other"), (r"\ball (received )?requests\b", "the requests"),
    (r"\bspecific zip code\b", "some place"), (r"\bchosen postal code\b", "some place"), (r"\bgiven zip code\b", "some place"),
]


def _apply(rules, text: str) -> str:
    for pat, rep in rules:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


def degrade_metric_text(text: str, level: str) -> str:
    """Rule-based degradation of the *textual* sentence of a metric prompt: informal | unclear | ambiguous."""
    if level == "informal":
        return _apply(_INFORMAL, text).lower()
    if level == "unclear":
        t = _apply(_VAGUE, text)
        t = re.split(r"[;,]\s*(?:i\.e\.|to |which|where|so )", t)[0]
        return t.rstrip(".") + "."
    if level == "ambiguous":
        return _apply(_AMBIG, _apply(_VAGUE[:6], text))
    raise ValueError("level must be informal, unclear or ambiguous")


def degraded_metric_prompt_set(ptype: str, level: str) -> list[MetricPrompt]:
    """C3 set: degrade ONLY the textual first line (formulae/pseudocode are kept, as in the paper)."""
    out = []
    for p in metric_prompt_set(ptype):
        first, _, rest = p.text.partition("\n")
        new = degrade_metric_text(first, level) + (("\n" + rest) if rest else "")
        out.append(MetricPrompt(p.prompt_id + f"-{level}", p.metric_id, ptype, new, f"rule_degraded:{level}", level))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Trade-off prompts (C2/C4/C6): Table 4 seeds (verbatim) + hand-written paraphrases
# ---------------------------------------------------------------------------------------------------------------
TRADEOFF_SEEDS: dict[int, str] = {
    1: "Maximize the number of accepted requests. Additionally, prioritize servicing new requests as soon as possible (e.g., today rather than tomorrow), even if it results in longer tours.",
    2: "Maximize the number of accepted requests. Furthermore, service new requests as soon as possible (e.g., today rather than tomorrow), but only if this does not result in longer tours.",
    3: "Maximize the number of accepted requests but reject tight time-window requests (e.g., requests requiring service at an exact time); this criterion aims to prevent overconstraining the fleet plan, which could hinder the acceptance of additional requests later.",
    5: "Maximize the number of accepted requests and prioritize servicing new requests as soon as possible (e.g., today rather than tomorrow). However, avoid servicing requests located in specific ZIP code zones (e.g., 00184, covering areas like the Colosseum, Roman Forum, and Monti in central Rome) if scheduled during peak hours (e.g., 11:30 a.m. - 1:30 p.m.), to reduce the risk of delays due to local congestion.",
    6: "Maximize the number of accepted requests and prioritize servicing new requests as soon as possible (e.g., today rather than tomorrow). However, maintaining route balance is essential—avoid overloading some vehicles while underutilizing others.",
    7: "Maximize the number of accepted requests, but reserve a portion of vehicle capacity for high-priority requests expected later in the day, based on historical patterns.",
    8: "Maximize the number of accepted requests while minimizing the number of vehicles returning to the depot after a specified time (e.g., 6:00 p.m.), to reduce overtime costs and respect driver schedules.",
    9: "Maximize the number of accepted requests while ensuring that no vehicle is assigned long routes on consecutive days, to prevent driver fatigue and promote fair workload distribution.",
    10: "Maximize accepted requests while promoting customer familiarity by minimizing the number of different drivers assigned to the same customer over a week.",
}

TRADEOFF_PARAPHRASES: dict[int, list[str]] = {
    1: ["Our first goal is to accept as many requests as we can; we also want new requests served today instead of tomorrow, even when that makes the tours longer.",
        "Accept the largest possible number of requests and favour same-day service of new requests, accepting longer tours if needed.",
        "Take as many requests as possible, and serve the new ones as early as we can (today before tomorrow) without worrying about tour length."],
    2: ["Accept as many requests as possible. New requests should be served as early as possible (today rather than tomorrow) provided the tours do not get longer.",
        "Maximise accepted requests; prefer serving new requests today instead of tomorrow only when this does not make the tours longer.",
        "The priority is the number of accepted requests, then early service of new requests, but never at the cost of longer tours."],
    3: ["Accept as many requests as possible, but turn down requests with tight time windows (such as an exact service time), since they overconstrain the plan and block later requests.",
        "Maximise acceptance while rejecting the requests that have very narrow time windows, to keep flexibility for future arrivals.",
        "We want a high number of accepted requests; however requests demanding service at an exact time should be rejected so they do not overconstrain the fleet."],
    5: ["Accept as many requests as possible and serve new requests today rather than tomorrow, but avoid visiting ZIP code 00184 around the Colosseum during the midday peak (11:30-13:30) to limit congestion delays.",
        "Maximise accepted requests, favour same-day service of new requests, and keep requests in the congested central-Rome ZIP zones out of the peak hours.",
        "Get the highest number of accepted requests with early service of new requests; however do not schedule service in the congested ZIP zones during peak hours."],
    6: ["Accept as many requests as possible and serve new requests as soon as possible, today rather than tomorrow; but keep the workload balanced so that no vehicle is overloaded while others are underused.",
        "Maximise accepted requests and favour same-day service of new requests, while keeping the routes balanced across vehicles.",
        "We want many accepted requests, served early; route balance matters too, so avoid overloading some vehicles and underutilizing others."],
    7: ["Accept as many requests as possible, but keep part of the vehicle capacity free for high-priority requests that, according to history, arrive later in the day.",
        "Maximise accepted requests while reserving some vehicle capacity for the high-priority requests we expect later.",
        "Our aim is a high number of accepted requests, but do not fill the vehicles completely: leave room for high-priority requests expected later."],
    8: ["Accept as many requests as possible while keeping as few vehicles as possible returning to the depot after 6:00 p.m., to cut overtime and respect driver schedules.",
        "Maximise accepted requests and reduce the number of vehicles that come back to the depot late (after 6 p.m.).",
        "We want many accepted requests but few late returns to the depot after 6 pm, because of overtime costs and driver schedules."],
    9: ["Accept as many requests as possible, but do not give any vehicle long routes on consecutive days, to avoid driver fatigue and share the workload fairly.",
        "Maximise accepted requests while making sure that no vehicle works long routes two days in a row.",
        "The goal is many accepted requests; however a driver should not get long routes on consecutive days (fatigue, fairness)."],
    10: ["Accept as many requests as possible and build customer familiarity by keeping the number of different drivers per customer low during the week.",
         "Maximise accepted requests while having each customer served by as few different drivers as possible over a week.",
         "We want many accepted requests, and also that the same customer sees the same driver again and again."],
}


@dataclass(frozen=True)
class TradeoffPrompt:
    prompt_id: str
    rule_no: int  # key of mcdm.reference_rules.RULES (paper Table 4 numbering)
    text: str
    source: str
    degradation_level: int = 0


def tradeoff_prompt_set(n: int = 30) -> list[TradeoffPrompt]:
    """30 trade-off prompts: the 9 Table-4 seeds, 18 hand-written paraphrases (2 per seed) and 3 further paraphrases
    (third variant of seeds 1, 2, 3). Paper: 30 distinct prompts; how many per seed is not stated."""
    out: list[TradeoffPrompt] = []
    for k, text in TRADEOFF_SEEDS.items():
        out.append(TradeoffPrompt(f"TO{k}-seed", k, text, "table4_seed"))
    for k in TRADEOFF_SEEDS:
        for i, t in enumerate(TRADEOFF_PARAPHRASES[k][:2], start=1):
            out.append(TradeoffPrompt(f"TO{k}-p{i}", k, t, "handwritten_paraphrase"))
    for k in (1, 2, 3):
        out.append(TradeoffPrompt(f"TO{k}-p3", k, TRADEOFF_PARAPHRASES[k][2], "handwritten_paraphrase"))
    return out[:n]


def degrade_tradeoff_text(text: str, level: int) -> str:
    """Rule-based progressive reduction of clarity (paper C4: LLM-made, NOT AVAILABLE).

    level 1: informal wording, parenthetical examples removed; level 2: additionally the qualifying clause after
    'however/but/while/to ...' is dropped; level 3: only the opening words are kept.
    """
    if level == 0:
        return text
    t = re.sub(r"\s*\([^)]*\)", "", text)
    t = _apply([(r"\bprioritize servicing\b", "try to serve"), (r"\bMaximi[sz]e the number of\b", "try to get lots of"), (r"\bMaximi[sz]e\b", "try to get lots of")], t)
    if level == 1:
        return t.lower()
    if level == 2:
        parts = re.split(r"\.\s+|[,;]\s*(?:however|but|while|to)\b|\s+(?:however|but|while)\s+", t, maxsplit=1)
        return (parts[0].rstrip(".") + ", whatever works.").lower()
    if level == 3:
        words = t.split()
        return " ".join(words[:7]).lower() + " ... you decide"
    raise ValueError("level must be 0-3")


def degraded_tradeoff_prompt_set(level: int) -> list[TradeoffPrompt]:
    return [TradeoffPrompt(p.prompt_id + f"-L{level}", p.rule_no, degrade_tradeoff_text(p.text, level), f"rule_degraded:L{level}", level) for p in tradeoff_prompt_set()]
