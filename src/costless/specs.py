"""Declarative scorer specifications, as written in costless.yaml and dataset files.

These are pure data. They are turned into executable scorers by
:func:`costless.scorers.build_scorer`.
"""

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


ScorerSpec = Annotated[
    ExactMatchSpec | RegexSpec | JsonSchemaSpec,
    Field(discriminator="type"),
]
