"""Small, dependency-free descriptive statistics used in run summaries."""

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def sample_variance(values: Sequence[float]) -> float:
    """Unbiased (n-1) variance; 0.0 for fewer than two values."""
    n = len(values)
    if n < 2:
        return 0.0
    m = mean(values)
    return math.fsum((v - m) ** 2 for v in values) / (n - 1)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (same as numpy's default), ``q`` in [0, 100]."""
    if not 0 <= q <= 100:
        msg = f"percentile must be in [0, 100], got {q}"
        raise ValueError(msg)
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
