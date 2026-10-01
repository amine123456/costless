"""Declarative scorer specifications, as written in costless.yaml and dataset files.

These are pure data. They are turned into executable scorers by
:func:`costless.scorers.build_scorer`.
"""

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _ScorerSpecBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    name: str | None = Field(default=None, description="Display name; defaults to the type.")
    weight: float = Field(default=1.0, gt=0, description="Weight in the case quality score.")


class ExactMatchSpec(_ScorerSpecBase):
    """Output (or a field of the JSON output) must equal the case's expected value."""

    type: Literal["exact_match"]
    path: str | None = Field(
        default=None,
        description="Dot path into the JSON output and the expected value, e.g. 'severity'.",
    )
    case_sensitive: bool = True
    normalize_whitespace: bool = True


class RegexSpec(_ScorerSpecBase):
    """Output (or a field of the JSON output) must match (or, if negated, not match) a pattern."""

    type: Literal["regex"]
    pattern: str
    path: str | None = None
    ignore_case: bool = False
    negate: bool = False

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            msg = f"invalid regex {value!r}: {exc}"
            raise ValueError(msg) from exc
        return value


class JsonSchemaSpec(_ScorerSpecBase):
    """Output must parse as JSON and validate against a JSON Schema (inline or a file path)."""

    type: Literal["json_schema"]
    json_schema: dict[str, Any] | str = Field(alias="schema")


class CalibrationSettings(BaseModel):
    """Human-labelled examples used by ``costless calibrate`` to vet this judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: str = Field(description="YAML/JSONL file of human-labelled examples.")
    min_pass_agreement: float | None = Field(default=None, ge=0, le=1)
    min_kappa: float | None = Field(default=None, ge=-1, le=1)


class LLMJudgeSpec(_ScorerSpecBase):
    """An LLM grades the output against a rubric on an integer scale."""

    type: Literal["llm_judge"]
    rubric: str | None = Field(
        default=None,
        description="Criteria for every case. Cases add their own `rubric` on top.",
    )
    scale_min: int = 1
    scale_max: int = 5
    anchors: dict[int, str] = Field(
        default_factory=dict, description="What each score means, e.g. {1: ..., 5: ...}."
    )
    pass_threshold: int | None = Field(
        default=None, description="Lowest passing score; defaults to scale_max - 1."
    )
    provider: str | None = Field(
        default=None, description="Judge provider; defaults to COSTLESS_JUDGE_PROVIDER."
    )
    model: str | None = Field(
        default=None, description="Judge model; defaults to COSTLESS_JUDGE_MODEL."
    )
    max_tokens: int = Field(default=2048, gt=0)
    include_reference: bool = Field(
        default=True, description="Show the case's `expected` value to the judge."
    )
    calibration: CalibrationSettings | None = None

    @model_validator(mode="after")
    def _check_scale(self) -> "LLMJudgeSpec":
        if self.scale_max <= self.scale_min:
            msg = "scale_max must be greater than scale_min"
            raise ValueError(msg)
        if not self.scale_min <= self.threshold <= self.scale_max:
            msg = "pass_threshold must lie within the scale"
            raise ValueError(msg)
        outside = [k for k in self.anchors if not self.scale_min <= k <= self.scale_max]
        if outside:
            msg = f"anchors outside the scale: {outside}"
            raise ValueError(msg)
        return self

    @property
    def threshold(self) -> int:
        return self.scale_max - 1 if self.pass_threshold is None else self.pass_threshold


ScorerSpec = Annotated[
    ExactMatchSpec | RegexSpec | JsonSchemaSpec | LLMJudgeSpec,
    Field(discriminator="type"),
]
