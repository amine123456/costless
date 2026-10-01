"""Aggregate attempts into per-case and per-run summaries."""

from collections.abc import Sequence

from costless.metrics import mean, percentile, sample_variance
from costless.models import Attempt, CaseSummary, RunSummary


def summarize_case(
    case_id: str, dataset: str, tags: tuple[str, ...], attempts: Sequence[Attempt]
) -> CaseSummary:
    qualities = [a.quality for a in attempts]
    passes = sum(a.passed for a in attempts)
    return CaseSummary(
        case_id=case_id,
        dataset=dataset,
        tags=tags,
        attempts=len(attempts),
        quality_mean=mean(qualities),
        quality_variance=sample_variance(qualities),
        pass_rate=passes / len(attempts),
        errors=sum(a.error is not None for a in attempts),
        latency_mean_ms=mean([a.latency_ms for a in attempts]),
        input_tokens_mean=mean([a.input_tokens for a in attempts]),
        output_tokens_mean=mean([a.output_tokens for a in attempts]),
        cost_mean_usd=_mean_cost(attempts),
        flaky=0 < passes < len(attempts),
    )


def summarize_run(
    cases: Sequence[CaseSummary],
    attempts: Sequence[Attempt],
    unpriced_models: Sequence[str] = (),
) -> RunSummary:
    executed = [a for a in attempts if not a.skipped]
    latencies = [a.latency_ms for a in executed]
    n = len(attempts)
    costs = [a.cost_usd for a in executed]
    priced = None if any(c is None for c in costs) else [c for c in costs if c is not None]
    eval_costs = [a.eval_cost_usd for a in executed]
    eval_total = None if any(c is None for c in eval_costs) else sum(c or 0.0 for c in eval_costs)
    return RunSummary(
        cases=len(cases),
        attempts=n,
        # Mean of case means: every case weighs the same regardless of repeats.
        quality_mean=mean([c.quality_mean for c in cases]),
        failure_rate=sum(not a.passed for a in attempts) / n if n else 0.0,
        error_rate=sum(a.error is not None for a in attempts) / n if n else 0.0,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        input_tokens=sum(a.input_tokens for a in attempts),
        output_tokens=sum(a.output_tokens for a in attempts),
        flaky_cases=sum(c.flaky for c in cases),
        cost_total_usd=None if priced is None else sum(priced),
        cost_per_case_usd=None if priced is None else mean(priced),
        eval_cost_total_usd=eval_total,
        unpriced_models=tuple(sorted(set(unpriced_models))),
        skipped_attempts=n - len(executed),
        scorer_errors=sum(r.error for a in attempts for r in a.scores),
    )


def _mean_cost(attempts: Sequence[Attempt]) -> float | None:
    costs = [a.cost_usd for a in attempts if not a.skipped]
    if not costs or any(c is None for c in costs):
        return None
    return mean([c for c in costs if c is not None])
