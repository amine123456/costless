"""Compare a candidate run with a baseline run and decide whether to block the merge.

Decision rule, per metric (details and rationale in docs/statistics.md):

- **regressed**: the change is statistically significant (the whole confidence
  interval of the difference, or ratio, lies on the bad side of "no change") *and*
  the point estimate is worse than the configured tolerance. Blocks the merge.
- **inconclusive**: the point estimate is worse than the tolerance, but the interval
  still includes "no change". Reported as a warning (blocks only with
  ``fail_on_inconclusive``); the usual remedy is more repeats or more cases.
- **improved**: significantly better.
- **ok**: none of the above.
- **breached**: an absolute limit (``min`` / ``max``) is violated by the candidate
  alone, whatever the baseline says. Blocks the merge.
- **unavailable**: not computable (e.g. costs could not be priced).

Without a baseline (the very first run on the main branch), only absolute limits apply.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from costless.config import GateSettings
from costless.models import Attempt, CaseSummary, RunResult, ScoreResult
from costless.stats import Statistic, paired_bootstrap, point_statistic

MAX_LISTED_CASES = 15


class Status(StrEnum):
    OK = "ok"
    IMPROVED = "improved"
    INCONCLUSIVE = "inconclusive"
    REGRESSED = "regressed"
    BREACHED = "breached"
    UNAVAILABLE = "unavailable"


BLOCKING = frozenset({Status.REGRESSED, Status.BREACHED})


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MetricComparison(_Model):
    metric: str
    kind: Literal["difference", "ratio"]
    better: Literal["higher", "lower"]
    baseline: float | None
    candidate: float | None
    change: float | None  # candidate - baseline, or candidate / baseline
    ci: tuple[float, float] | None
    tolerance: float | None  # absolute (difference) or relative (ratio - 1)
    limit: float | None  # absolute min/max on the candidate
    status: Status
    note: str | None = None


class CaseChange(_Model):
    case_id: str
    baseline_quality: float
    candidate_quality: float
    baseline_pass_rate: float
    candidate_pass_rate: float
    example_failure: str | None


class Comparison(_Model):
    baseline_ref: str | None
    baseline_sha: str | None
    candidate_ref: str | None
    candidate_sha: str | None
    confidence: float
    bootstrap_samples: int
    paired_cases: int
    repeats: tuple[int | None, int]
    only_in_baseline: tuple[str, ...]
    only_in_candidate: tuple[str, ...]
    datasets_changed: bool
    metrics: tuple[MetricComparison, ...]
    regressed_cases: tuple[CaseChange, ...]
    fixed_cases: tuple[CaseChange, ...]
    gate_passed: bool
    reasons: tuple[str, ...]


def compare_runs(
    candidate: RunResult, baseline: RunResult | None, gate: GateSettings
) -> Comparison:
    if baseline is None:
        metrics = _absolute_only(candidate, gate)
        paired: list[str] = []
        only_base: tuple[str, ...] = ()
        only_cand: tuple[str, ...] = ()
        regressed: tuple[CaseChange, ...] = ()
        fixed: tuple[CaseChange, ...] = ()
        datasets_changed = False
    else:
        base_ids = {c.case_id for c in baseline.cases}
        cand_ids = {c.case_id for c in candidate.cases}
        paired = [c.case_id for c in candidate.cases if c.case_id in base_ids]
        only_base = tuple(sorted(base_ids - cand_ids))
        only_cand = tuple(sorted(cand_ids - base_ids))
        datasets_changed = {d.sha256 for d in baseline.metadata.datasets} != {
            d.sha256 for d in candidate.metadata.datasets
        }
        metrics = (
            _paired_metrics(candidate, baseline, paired, gate)
            if paired
            else (_absolute_only(candidate, gate))
        )
        regressed, fixed = _case_changes(candidate, baseline, paired, gate)

    reasons = [
        f"{m.metric}: {m.status.value}" + (f" ({m.note})" if m.note else "")
        for m in metrics
        if m.status in BLOCKING or (gate.fail_on_inconclusive and m.status == Status.INCONCLUSIVE)
    ]
    return Comparison(
        baseline_ref=baseline.metadata.git_ref if baseline else None,
        baseline_sha=baseline.metadata.git_sha if baseline else None,
        candidate_ref=candidate.metadata.git_ref,
        candidate_sha=candidate.metadata.git_sha,
        confidence=gate.confidence,
        bootstrap_samples=gate.bootstrap_samples,
        paired_cases=len(paired),
        repeats=(baseline.metadata.repeats if baseline else None, candidate.metadata.repeats),
        only_in_baseline=only_base,
        only_in_candidate=only_cand,
        datasets_changed=datasets_changed,
        metrics=tuple(metrics),
        regressed_cases=regressed,
        fixed_cases=fixed,
        gate_passed=not reasons,
        reasons=tuple(reasons),
    )


# --------------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class _Pair:
    candidate: RunResult
    baseline: RunResult
    paired: list[str]
    gate: GateSettings


Extractor = Callable[[Attempt], float | None]


def _quality(a: Attempt) -> float | None:
    return a.quality


def _failure(a: Attempt) -> float | None:
    return 0.0 if a.passed else 1.0


def _cost(a: Attempt) -> float | None:
    return None if a.skipped else a.cost_usd


def _latency(a: Attempt) -> float | None:
    return None if a.skipped or a.error else a.latency_ms


def _paired_metrics(
    candidate: RunResult, baseline: RunResult, paired: list[str], gate: GateSettings
) -> list[MetricComparison]:
    q, f, c, lat = gate.quality, gate.failure_rate, gate.cost_per_case, gate.latency_p95
    pair = _Pair(candidate, baseline, paired, gate)
    summary = candidate.summary
    return [
        _paired_difference(
            pair,
            "quality",
            "higher",
            _quality,
            tolerance=q.max_drop,
            limit=q.min,
            candidate_value=summary.quality_mean,
        ),
        _paired_difference(
            pair,
            "failure_rate",
            "lower",
            _failure,
            tolerance=f.max_increase,
            limit=f.max,
            candidate_value=summary.failure_rate,
        ),
        _paired_ratio(
            pair,
            "cost_per_case",
            "mean",
            _cost,
            tolerance_pct=c.max_increase_pct,
            limit=None,
            candidate_value=summary.cost_per_case_usd,
        ),
        _paired_ratio(
            pair,
            "latency_p95_ms",
            "p95",
            _latency,
            tolerance_pct=lat.max_increase_pct,
            limit=lat.max_ms,
            candidate_value=summary.latency_p95_ms,
        ),
    ]


def _values(
    run: RunResult, paired: list[str], extract: Extractor
) -> list[NDArray[np.float64]] | None:
    by_case: dict[str, list[float]] = {cid: [] for cid in paired}
    for attempt in run.attempts:
        if attempt.case_id in by_case:
            value = extract(attempt)
            if value is not None:
                by_case[attempt.case_id].append(value)
    arrays = [np.asarray(by_case[cid], dtype=np.float64) for cid in paired]
    return None if any(len(a) == 0 for a in arrays) else arrays


def _paired_difference(
    pair: _Pair,
    name: str,
    better: Literal["higher", "lower"],
    extract: Extractor,
    *,
    tolerance: float | None,
    limit: float | None,
    candidate_value: float,
) -> MetricComparison:
    gate = pair.gate
    base_vals = _values(pair.baseline, pair.paired, extract)
    cand_vals = _values(pair.candidate, pair.paired, extract)
    breached = _breached(candidate_value, limit, better)
    if base_vals is None or cand_vals is None:
        return _unavailable(
            name,
            kind="difference",
            better=better,
            candidate_value=candidate_value,
            tolerance=tolerance,
            limit=limit,
            breached=breached,
        )

    base_point = point_statistic(base_vals, "mean")
    cand_point = point_statistic(cand_vals, "mean")
    change = cand_point - base_point
    boot = paired_bootstrap(
        base_vals, cand_vals, statistic="mean", samples=gate.bootstrap_samples, seed=gate.seed
    )
    low, high = boot.difference_ci(gate.confidence)
    # Orient the interval so that "worse" is always negative. Flipping the sign of
    # a lower-is-better metric also swaps which bound is the upper one.
    sign = 1.0 if better == "higher" else -1.0
    oriented_low, oriented_high = sorted((sign * low, sign * high))
    status = _decide(
        worse_significant=oriented_high < 0,
        better_significant=oriented_low > 0,
        worse_than_tolerance=tolerance is not None and sign * change < -tolerance,
        breached=breached,
    )
    return MetricComparison(
        metric=name,
        kind="difference",
        better=better,
        baseline=base_point,
        candidate=cand_point,
        change=change,
        ci=(low, high),
        tolerance=tolerance,
        limit=limit,
        status=status,
        note=_note(status, limit, candidate_value),
    )


def _paired_ratio(
    pair: _Pair,
    name: str,
    statistic: Statistic,
    extract: Extractor,
    *,
    tolerance_pct: float | None,
    limit: float | None,
    candidate_value: float | None,
) -> MetricComparison:
    gate = pair.gate
    tolerance = None if tolerance_pct is None else tolerance_pct / 100
    breached = _breached(candidate_value, limit, "lower")

    def unavailable() -> MetricComparison:
        return _unavailable(
            name,
            kind="ratio",
            better="lower",
            candidate_value=candidate_value,
            tolerance=tolerance,
            limit=limit,
            breached=breached,
        )

    base_vals = _values(pair.baseline, pair.paired, extract)
    cand_vals = _values(pair.candidate, pair.paired, extract)
    if base_vals is None or cand_vals is None:
        return unavailable()
    base_point = point_statistic(base_vals, statistic)
    cand_point = point_statistic(cand_vals, statistic)
    if base_point <= 0:
        return unavailable()

    change = cand_point / base_point
    boot = paired_bootstrap(
        base_vals, cand_vals, statistic=statistic, samples=gate.bootstrap_samples, seed=gate.seed
    )
    ci = boot.ratio_ci(gate.confidence)
    if ci is None:
        return unavailable()
    low, high = ci
    status = _decide(
        worse_significant=low > 1,
        better_significant=high < 1,
        worse_than_tolerance=tolerance is not None and change > 1 + tolerance,
        breached=breached,
    )
    return MetricComparison(
        metric=name,
        kind="ratio",
        better="lower",
        baseline=base_point,
        candidate=cand_point,
        change=change,
        ci=(low, high),
        tolerance=tolerance,
        limit=limit,
        status=status,
        note=_note(status, limit, candidate_value),
    )


def _decide(
    *,
    worse_significant: bool,
    better_significant: bool,
    worse_than_tolerance: bool,
    breached: bool,
) -> Status:
    if breached:
        return Status.BREACHED
    if worse_significant and worse_than_tolerance:
        return Status.REGRESSED
    if worse_than_tolerance:
        return Status.INCONCLUSIVE
    if better_significant:
        return Status.IMPROVED
    return Status.OK


def _absolute_only(candidate: RunResult, gate: GateSettings) -> list[MetricComparison]:
    s = candidate.summary
    rows: list[
        tuple[
            str,
            Literal["difference", "ratio"],
            Literal["higher", "lower"],
            float | None,
            float | None,
        ]
    ] = [
        ("quality", "difference", "higher", s.quality_mean, gate.quality.min),
        ("failure_rate", "difference", "lower", s.failure_rate, gate.failure_rate.max),
        ("cost_per_case", "ratio", "lower", s.cost_per_case_usd, None),
        ("latency_p95_ms", "ratio", "lower", s.latency_p95_ms, gate.latency_p95.max_ms),
    ]
    out = []
    for name, kind, better, value, limit in rows:
        breached = _breached(value, limit, better)
        status = Status.BREACHED if breached else Status.OK
        out.append(
            MetricComparison(
                metric=name,
                kind=kind,
                better=better,
                baseline=None,
                candidate=value,
                change=None,
                ci=None,
                tolerance=None,
                limit=limit,
                status=status,
                note=_note(status, limit, value) or "no baseline: absolute limits only",
            )
        )
    return out


def _breached(value: float | None, limit: float | None, better: str) -> bool:
    if value is None or limit is None:
        return False
    return value < limit if better == "higher" else value > limit


def _unavailable(
    name: str,
    *,
    kind: Literal["difference", "ratio"],
    better: Literal["higher", "lower"],
    candidate_value: float | None,
    tolerance: float | None,
    limit: float | None,
    breached: bool,
) -> MetricComparison:
    status = Status.BREACHED if breached else Status.UNAVAILABLE
    return MetricComparison(
        metric=name,
        kind=kind,
        better=better,
        baseline=None,
        candidate=candidate_value,
        change=None,
        ci=None,
        tolerance=tolerance,
        limit=limit,
        status=status,
        note=_note(status, limit, candidate_value) or "missing data in one of the runs",
    )


def _note(status: Status, limit: float | None, value: float | None) -> str | None:
    if status == Status.BREACHED and value is not None and limit is not None:
        return f"{value:.4g} violates the absolute limit {limit:.4g}"
    return None


# --------------------------------------------------------------------------- cases


def _case_changes(
    candidate: RunResult, baseline: RunResult, paired: list[str], gate: GateSettings
) -> tuple[tuple[CaseChange, ...], tuple[CaseChange, ...]]:
    """Cases that got worse (or better). Individually these are not significance-
    tested: they explain an aggregate result, they do not decide it."""
    base = {c.case_id: c for c in baseline.cases}
    cand = {c.case_id: c for c in candidate.cases}
    tolerance = gate.quality.max_drop or 0.0
    worse: list[CaseChange] = []
    better: list[CaseChange] = []
    for case_id in paired:
        b, c = base[case_id], cand[case_id]
        change = _case_change(case_id, b, c, candidate)
        delta = c.quality_mean - b.quality_mean
        if c.pass_rate < b.pass_rate or delta < -tolerance:
            worse.append(change)
        elif c.pass_rate > b.pass_rate or delta > tolerance:
            better.append(change)
    worse.sort(key=lambda x: (x.candidate_quality - x.baseline_quality, x.case_id))
    better.sort(key=lambda x: (x.baseline_quality - x.candidate_quality, x.case_id))
    return tuple(worse[:MAX_LISTED_CASES]), tuple(better[:MAX_LISTED_CASES])


def _case_change(case_id: str, b: CaseSummary, c: CaseSummary, run: RunResult) -> CaseChange:
    return CaseChange(
        case_id=case_id,
        baseline_quality=b.quality_mean,
        candidate_quality=c.quality_mean,
        baseline_pass_rate=b.pass_rate,
        candidate_pass_rate=c.pass_rate,
        example_failure=_first_failure(run, case_id),
    )


def _first_failure(run: RunResult, case_id: str) -> str | None:
    for attempt in run.attempts:
        if attempt.case_id != case_id or attempt.passed:
            continue
        if attempt.error:
            return attempt.error
        failed: list[ScoreResult] = [s for s in attempt.scores if not s.passed]
        if failed:
            first = failed[0]
            return f"{first.scorer}: {first.detail or 'failed'}"
    return None
