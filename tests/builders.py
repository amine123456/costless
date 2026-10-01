"""Build synthetic run.json documents for comparison tests."""

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from costless.models import Attempt, DatasetInfo, RunMetadata, RunResult, ScoreResult
from costless.summarize import summarize_case, summarize_run


def make_run(
    qualities: Mapping[str, Sequence[float]],
    *,
    latency_ms: float | Mapping[str, Sequence[float]] = 100.0,
    cost_usd: float | None = 0.001,
    pass_at: float = 0.5,
    ref: str = "feature",
    sha: str = "c" * 40,
    dataset_sha: str = "d" * 64,
) -> RunResult:
    attempts: list[Attempt] = []
    for case_id, values in qualities.items():
        for repeat, quality in enumerate(values):
            passed = quality >= pass_at
            latency = (
                latency_ms if isinstance(latency_ms, float | int) else latency_ms[case_id][repeat]
            )
            attempts.append(
                Attempt(
                    case_id=case_id,
                    repeat=repeat,
                    output="out",
                    error=None,
                    latency_ms=float(latency),
                    scores=(
                        ScoreResult(
                            scorer="judge",
                            score=quality,
                            passed=passed,
                            detail=None if passed else f"{case_id} | wrong answer",
                        ),
                    ),
                    quality=quality,
                    passed=passed,
                    cost_usd=cost_usd,
                )
            )
    cases = tuple(
        summarize_case(cid, "ds", (), [a for a in attempts if a.case_id == cid])
        for cid in qualities
    )
    repeats = max(len(v) for v in qualities.values())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return RunResult(
        metadata=RunMetadata(
            run_id="r",
            started_at=now,
            finished_at=now,
            costless_version="0",
            git_sha=sha,
            git_ref=ref,
            config_sha256="x",
            target="t",
            repeats=repeats,
            datasets=(DatasetInfo(name="ds", version=None, sha256=dataset_sha, cases=len(cases)),),
        ),
        summary=summarize_run(cases, attempts),
        cases=cases,
        attempts=tuple(attempts),
    )
