"""Execute every case N times against the target and score each attempt.

Failure semantics, used throughout costless:

- An attempt *errors* when the target raises, times out or returns unusable output.
  An errored attempt scores 0 and is not passed.
- An attempt *passes* when it did not error and every scorer passed.
- ``failure_rate`` is the share of attempts that did not pass (errors included).
- ``quality`` is the weight-averaged score of all scorers, in [0, 1].
"""

import asyncio
import logging
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from costless import __version__
from costless.config import LoadedConfig
from costless.context import attempt_scope
from costless.dataset import Dataset, load_dataset
from costless.errors import ConfigError
from costless.models import Attempt, Case, RunMetadata, RunResult, ScoreResult
from costless.pricing import to_report
from costless.scorers import Scorer, build_scorer
from costless.summarize import summarize_case, summarize_run
from costless.targets import Target, TargetResponse, build_target

log = logging.getLogger(__name__)

ProgressCallback = Callable[[Attempt], None]


@dataclass(frozen=True)
class WorkItem:
    """A case bound to its dataset and to the scorers that grade it."""

    case_id: str
    dataset: str
    case: Case
    scorers: tuple[Scorer, ...]


def prepare(
    loaded: LoadedConfig, *, tags: Sequence[str] = ()
) -> tuple[list[Dataset], list[WorkItem]]:
    """Load datasets and build scorers; fail fast on anything that cannot be graded."""
    config = loaded.config
    defaults = tuple(build_scorer(s, loaded.base_dir) for s in config.scorers)

    datasets: list[Dataset] = []
    items: list[WorkItem] = []
    problems: list[str] = []
    seen_names: set[str] = set()
    for ref in config.datasets:
        dataset = load_dataset(loaded.resolve(ref.path)).filter_tags((*ref.tags, *tags))
        if dataset.name in seen_names:
            msg = f"two datasets are named {dataset.name!r}; dataset names must be unique"
            raise ConfigError(msg)
        seen_names.add(dataset.name)
        datasets.append(dataset)
        for case in dataset.cases:
            own = tuple(build_scorer(s, dataset.path.parent) for s in case.scorers)
            scorers = defaults + own
            case_id = f"{dataset.name}/{case.id}"
            if not scorers:
                problems.append(f"{case_id}: no scorers configured")
            problems.extend(
                f"{case_id}: {problem}"
                for scorer in scorers
                if (problem := scorer.check_case(case)) is not None
            )
            items.append(WorkItem(case_id, dataset.name, case, scorers))

    if problems:
        raise ConfigError("cannot grade every case:\n  " + "\n  ".join(problems))
    if not items:
        raise ConfigError("no cases selected (check the tag filters)")
    return datasets, items


async def run_suite(
    loaded: LoadedConfig,
    *,
    repeats: int | None = None,
    tags: Sequence[str] = (),
    target: Target | None = None,
    on_attempt: ProgressCallback | None = None,
) -> RunResult:
    config = loaded.config
    n_repeats = repeats if repeats is not None else config.run.repeats
    if n_repeats < 1:
        msg = "repeats must be at least 1"
        raise ConfigError(msg)
    datasets, items = prepare(loaded, tags=tags)
    target = target or build_target(loaded)
    pricing = loaded.pricing_table()
    spend = _SpendGuard(config.budget.max_run_usd)
    unpriced: set[str] = set()

    started_at = datetime.now(UTC)
    semaphore = asyncio.Semaphore(config.run.concurrency)

    async def bounded(item: WorkItem, repeat: int) -> Attempt:
        async with semaphore:
            if spend.exhausted:
                attempt = _skipped(item, repeat, spend.limit_usd)
            else:
                attempt = await run_attempt(target, item, repeat)
                target_cost = pricing.cost(attempt.usage)
                eval_cost = pricing.cost(attempt.eval_usage)
                unpriced.update(target_cost.unpriced_models, eval_cost.unpriced_models)
                spend.add(target_cost.usd + eval_cost.usd)  # the wallet pays for both
                attempt = attempt.model_copy(
                    update={
                        "cost_usd": to_report(target_cost.usd) if target_cost.complete else None,
                        "eval_cost_usd": to_report(eval_cost.usd) if eval_cost.complete else None,
                    }
                )
        if on_attempt is not None:
            on_attempt(attempt)
        return attempt

    attempts = await asyncio.gather(*(bounded(item, r) for item in items for r in range(n_repeats)))

    by_case: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        by_case.setdefault(attempt.case_id, []).append(attempt)
    summaries = tuple(
        summarize_case(item.case_id, item.dataset, item.case.tags, by_case[item.case_id])
        for item in items
    )
    git_sha, git_ref = _git_info(loaded.base_dir)
    metadata = RunMetadata(
        run_id=uuid.uuid4().hex,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        costless_version=__version__,
        git_sha=git_sha,
        git_ref=git_ref,
        config_sha256=loaded.sha256,
        target=target.name,
        repeats=n_repeats,
        datasets=tuple(d.info() for d in datasets),
    )
    return RunResult(
        metadata=metadata,
        summary=summarize_run(summaries, attempts, sorted(unpriced)),
        cases=summaries,
        attempts=tuple(attempts),
    )


async def run_attempt(target: Target, item: WorkItem, repeat: int) -> Attempt:
    response: TargetResponse | None = None
    error: str | None = None
    with attempt_scope(item.case_id, repeat) as scope:
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(target.invoke(item.case), target.timeout_s)
        except TimeoutError:
            error = f"timed out after {target.timeout_s:g}s"
        except Exception as exc:  # noqa: BLE001 - any target failure is a recorded attempt error
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        usage = (*scope.usage, *(response.usage if response is not None else ()))

    if response is None:
        return Attempt(
            case_id=item.case_id,
            repeat=repeat,
            output=None,
            error=error,
            latency_ms=latency_ms,
            usage=usage,
            quality=0.0,
            passed=False,
        )

    output = response.output
    # Scorers run in their own scope: LLM-judge calls are metered as evaluation
    # cost, never as cost of the system under test.
    with attempt_scope(f"{item.case_id}#eval", repeat) as eval_scope:
        scores = tuple([await _safe_score(s, item.case, output) for s in item.scorers])
    total_weight = sum(s.weight for s in item.scorers)
    quality = (
        sum(r.score * s.weight for r, s in zip(scores, item.scorers, strict=True)) / total_weight
    )
    return Attempt(
        case_id=item.case_id,
        repeat=repeat,
        output=output,
        error=None,
        latency_ms=latency_ms,
        usage=usage,
        scores=scores,
        eval_usage=tuple(eval_scope.usage),
        quality=min(1.0, max(0.0, quality)),
        passed=all(r.passed for r in scores),
    )


class _SpendGuard:
    """Tracks priced spend and trips once the run budget is used up.

    Attempts already in flight finish; attempts not yet started are skipped.
    Unpriced calls count as $0 here, so with lenient pricing this is a lower bound.
    """

    def __init__(self, limit_usd: float | None) -> None:
        self.limit_usd = limit_usd
        self._limit = None if limit_usd is None else Decimal(str(limit_usd))
        self._spent = Decimal(0)

    @property
    def exhausted(self) -> bool:
        return self._limit is not None and self._spent >= self._limit

    def add(self, usd: Decimal) -> None:
        self._spent += usd


def _skipped(item: WorkItem, repeat: int, limit_usd: float | None) -> Attempt:
    return Attempt(
        case_id=item.case_id,
        repeat=repeat,
        output=None,
        error=f"skipped: run budget of ${limit_usd:g} reached",
        latency_ms=0.0,
        quality=0.0,
        passed=False,
        skipped=True,
    )


async def _safe_score(scorer: Scorer, case: Case, output: str) -> ScoreResult:
    try:
        return await scorer.score(case, output)
    except Exception as exc:  # noqa: BLE001 - a crashing scorer must not abort the run
        log.warning("scorer %s crashed on %s: %s", scorer.name, case.id, exc)
        return ScoreResult(
            scorer=scorer.name, score=0.0, passed=False, detail=f"scorer error: {exc}", error=True
        )


def _git_info(cwd: Path) -> tuple[str | None, str | None]:
    """Commit and ref of the code under test, preferring CI-provided values."""
    sha = os.environ.get("CI_COMMIT_SHA") or os.environ.get("GITHUB_SHA")
    ref = (
        os.environ.get("CI_MERGE_REQUEST_SOURCE_BRANCH_NAME")
        or os.environ.get("CI_COMMIT_REF_NAME")
        or os.environ.get("GITHUB_HEAD_REF")
        or os.environ.get("GITHUB_REF_NAME")
    )
    if sha is None:
        sha = _git(cwd, "rev-parse", "HEAD")
    if ref is None:
        ref = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    return sha, ref


def _git(cwd: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],  # noqa: S607 - git is resolved from PATH on purpose
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None
