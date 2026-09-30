"""Core data model: eval cases, per-attempt results and the run.json document."""

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from costless.specs import ScorerSpec

RUN_SCHEMA_VERSION: Final = 1

CASE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Case(_Model):
    """One eval case: an input for the system under test plus what a good answer looks like."""

    id: str = Field(pattern=CASE_ID_PATTERN)
    input: Any
    expected: Any = None
    rubric: str | None = None
    tags: tuple[str, ...] = ()
    scorers: tuple[ScorerSpec, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(_Model):
    """Token usage of a single model call."""

    provider: str = "unknown"
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)


class ScoreResult(_Model):
    scorer: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    detail: str | None = None


class Attempt(_Model):
    """One execution of one case. A case is attempted ``repeats`` times per run."""

    case_id: str
    repeat: int = Field(ge=0)
    output: str | None
    error: str | None
    latency_ms: float = Field(ge=0)
    usage: tuple[Usage, ...] = ()
    scores: tuple[ScoreResult, ...] = ()
    quality: float = Field(ge=0.0, le=1.0)
    passed: bool

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usage)

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usage)


class CaseSummary(_Model):
    case_id: str
    dataset: str
    tags: tuple[str, ...]
    attempts: int
    quality_mean: float
    quality_variance: float
    pass_rate: float
    errors: int
    latency_mean_ms: float
    input_tokens_mean: float
    output_tokens_mean: float
    flaky: bool = Field(description="Passed on some repeats and failed on others.")


class RunSummary(_Model):
    cases: int
    attempts: int
    quality_mean: float
    failure_rate: float
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    input_tokens: int
    output_tokens: int
    flaky_cases: int


class DatasetInfo(_Model):
    name: str
    version: str | None
    sha256: str
    cases: int


class RunMetadata(_Model):
    run_id: str
    started_at: datetime
    finished_at: datetime
    costless_version: str
    git_sha: str | None
    git_ref: str | None
    config_sha256: str
    target: str
    repeats: int
    datasets: tuple[DatasetInfo, ...]


class RunResult(_Model):
    """The run.json document. Stores every attempt so any run can be re-analysed later."""

    schema_version: Literal[1] = RUN_SCHEMA_VERSION
    metadata: RunMetadata
    summary: RunSummary
    cases: tuple[CaseSummary, ...]
    attempts: tuple[Attempt, ...]
