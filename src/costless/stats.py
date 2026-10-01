"""Paired cluster bootstrap for comparing two runs.

Both runs evaluated the same cases, each case several times. Two sources of
noise matter:

1. *Which cases* happen to be in the dataset (between-case variance).
2. *Which answers* the model happened to give on this run (within-case
   variance, the reason every case is repeated).

Each bootstrap replicate resamples *cases* with replacement, using the same
cases for both runs (the comparison is paired), and keeps every attempt of a
chosen case. The within-case noise is already contained in each case's
observed attempts, so it propagates into the replicate without being resampled
again. Resampling the repeats as well (a two-stage bootstrap) counts that noise
twice and makes intervals too wide; this was measured on simulated runs and is
covered by the tests.

Replicates are processed in chunks to bound memory, and the generator is seeded,
so a comparison is reproducible bit for bit.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Statistic = Literal["mean", "p95"]
_CHUNK = 1000


@dataclass(frozen=True)
class BootstrapResult:
    base: NDArray[np.float64]
    candidate: NDArray[np.float64]

    def difference_ci(self, confidence: float) -> tuple[float, float]:
        return _percentile_ci(self.candidate - self.base, confidence)

    def ratio_ci(self, confidence: float) -> tuple[float, float] | None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = self.candidate / self.base
        if not np.all(np.isfinite(ratio)):
            return None
        return _percentile_ci(ratio, confidence)


def point_statistic(cases: Sequence[NDArray[np.float64]], statistic: Statistic) -> float:
    """The statistic on the observed data, matching what run summaries report."""
    if statistic == "mean":
        return float(np.mean([c.mean() for c in cases]))
    return float(np.percentile(np.concatenate(cases), 95))


def paired_bootstrap(
    base: Sequence[NDArray[np.float64]],
    candidate: Sequence[NDArray[np.float64]],
    *,
    statistic: Statistic,
    samples: int,
    seed: int,
) -> BootstrapResult:
    """Bootstrap ``statistic`` for both runs. ``base[i]`` and ``candidate[i]`` are the
    attempt values of case i in each run; every case needs at least one attempt in both.
    """
    if len(base) != len(candidate) or not base:
        msg = "need the same, non-zero number of cases in both runs"
        raise ValueError(msg)
    if any(len(c) == 0 for c in (*base, *candidate)):
        msg = "every case needs at least one attempt in both runs"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    base_matrix = _padded(base)
    cand_matrix = _padded(candidate)
    out_base = np.empty(samples)
    out_cand = np.empty(samples)
    n_cases = len(base)
    for start in range(0, samples, _CHUNK):
        b = min(_CHUNK, samples - start)
        chosen = rng.integers(0, n_cases, size=(b, n_cases))  # shared: paired design
        out_base[start : start + b] = _replicate(base_matrix, chosen, statistic)
        out_cand[start : start + b] = _replicate(cand_matrix, chosen, statistic)
    return BootstrapResult(base=out_base, candidate=out_cand)


@dataclass(frozen=True)
class _Padded:
    values: NDArray[np.float64]  # (cases, max_repeats), padded with NaN
    case_means: NDArray[np.float64]  # (cases,)
    uneven: bool  # some cases have fewer attempts than others


def _padded(cases: Sequence[NDArray[np.float64]]) -> _Padded:
    width = max(len(c) for c in cases)
    values = np.full((len(cases), width), np.nan)
    for i, c in enumerate(cases):
        values[i, : len(c)] = c
    return _Padded(
        values=values,
        case_means=np.array([c.mean() for c in cases]),
        uneven=len({len(c) for c in cases}) > 1,
    )


def _replicate(
    data: _Padded, chosen: NDArray[np.int64], statistic: Statistic
) -> NDArray[np.float64]:
    """One statistic per replicate row, over all attempts of the chosen cases."""
    resampled = data.values[chosen]  # (b, n_cases, max_repeats), NaN-padded
    if statistic == "mean":
        case_means = data.case_means[chosen]
        return np.asarray(case_means.mean(axis=1), dtype=np.float64)
    pooled = resampled.reshape(resampled.shape[0], -1)
    if data.uneven:  # the NaN-aware percentile is much slower; only pay for it when needed
        return np.asarray(np.nanpercentile(pooled, 95, axis=1), dtype=np.float64)
    return np.asarray(np.percentile(pooled, 95, axis=1), dtype=np.float64)


def _percentile_ci(values: NDArray[np.float64], confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)
