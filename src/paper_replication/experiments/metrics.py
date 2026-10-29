"""Small aggregation helpers for the experiments."""

from __future__ import annotations

from typing import Iterable, Optional


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the available values; None if none is available (never 0 as a stand-in)."""
    vs = [v for v in values if v is not None and v == v]  # v == v is False for NaN (pandas turns None into NaN)
    return (sum(vs) / len(vs)) if vs else None


def sum_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vs = list(values)
    if not vs or any(v is None or v != v for v in vs):
        return None
    return float(sum(vs))


def rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den
