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


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; well-behaved for small n and p near 0/1."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    """Unweighted Cohen's kappa. None when chance agreement is 1 (no variety at all)."""
    return weighted_kappa(a, b, categories=sorted(set(a) | set(b)), weights="none")


def weighted_kappa(
    a: Sequence[int],
    b: Sequence[int],
    *,
    categories: Sequence[int],
    weights: str = "quadratic",
) -> float | None:
    """Cohen's kappa with "none", "linear" or "quadratic" disagreement weights.

    Returns None when the expected disagreement is zero, i.e. kappa is undefined.
    """
    if len(a) != len(b):
        msg = "rating sequences must have the same length"
        raise ValueError(msg)
    if not a:
        return None
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    n = len(a)

    def weight(i: int, j: int) -> float:
        if weights == "none":
            return 0.0 if i == j else 1.0
        span = max(k - 1, 1)
        if weights == "linear":
            return abs(i - j) / span
        if weights == "quadratic":
            return ((i - j) / span) ** 2
        msg = f"unknown weights {weights!r}"
        raise ValueError(msg)

    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1 / n
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    disagreement_observed = math.fsum(
        weight(i, j) * observed[i][j] for i in range(k) for j in range(k)
    )
    disagreement_expected = math.fsum(
        weight(i, j) * row[i] * col[j] for i in range(k) for j in range(k)
    )
    if disagreement_expected == 0:
        return None
    return 1 - disagreement_observed / disagreement_expected
