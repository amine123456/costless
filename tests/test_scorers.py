import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import TypeAdapter, ValidationError

from costless.errors import ConfigError
from costless.models import ScoreResult
from costless.scorers import build_scorer
from costless.scorers._json import PathNotFoundError, get_path, parse_json_output
from costless.specs import ScorerSpec
from tests.conftest import make_case

SPEC: TypeAdapter[ScorerSpec] = TypeAdapter(ScorerSpec)


def score(
    spec: Mapping[str, object], output: str, *, base_dir: Path = Path(), **case: object
) -> ScoreResult:
    scorer = build_scorer(SPEC.validate_python(spec), base_dir)
    return asyncio.run(scorer.score(make_case(**case), output))


class TestJsonHelpers:
    def test_parses_fenced_json(self) -> None:
        assert parse_json_output('```json\n{"a": 1}\n```') == {"a": 1}

    def test_parses_plain_json_with_whitespace(self) -> None:
        assert parse_json_output('  {"a": [1, 2]}\n') == {"a": [1, 2]}

    def test_rejects_non_json(self) -> None:
        with pytest.raises(ValueError, match="Expecting value"):
            parse_json_output("not json")

    def test_get_path_through_dicts_and_lists(self) -> None:
        doc = {"actions": [{"owner": "db"}, {"owner": "net"}]}
        assert get_path(doc, "actions.1.owner") == "net"
        assert get_path(doc, "actions.-1.owner") == "net"

    @pytest.mark.parametrize("path", ["missing", "actions.5", "actions.x", "actions.0.owner.deep"])
    def test_get_path_missing(self, path: str) -> None:
        with pytest.raises(PathNotFoundError):
            get_path({"actions": [{"owner": "db"}]}, path)


class TestExactMatch:
    def test_string_match_normalises_whitespace_by_default(self) -> None:
        result = score({"type": "exact_match"}, "  hello \n world ", expected="hello world")
        assert result.passed
        assert result.score == 1.0

    def test_case_sensitive_by_default(self) -> None:
        assert not score({"type": "exact_match"}, "Hello", expected="hello").passed

    def test_case_insensitive(self) -> None:
        spec = {"type": "exact_match", "case_sensitive": False}
        assert score(spec, "HELLO", expected="hello").passed

    def test_mismatch_reports_both_values(self) -> None:
        result = score({"type": "exact_match"}, "b", expected="a")
        assert not result.passed
        assert result.score == 0.0
        assert result.detail == "expected 'a', got 'b'"

    def test_structured_expected_compares_parsed_json(self) -> None:
        expected = {"severity": "SEV1", "services": ["db"]}
        output = json.dumps({"services": ["db"], "severity": "SEV1"})
        assert score({"type": "exact_match"}, output, expected=expected).passed

    def test_path_compares_one_field(self) -> None:
        spec = {"type": "exact_match", "path": "severity"}
        output = '{"severity": "SEV2", "summary": "anything"}'
        result = score(spec, output, expected={"severity": "SEV2", "summary": "ignored"})
        assert result.passed
        assert result.scorer == "exact_match[severity]"

    def test_path_with_scalar_expected(self) -> None:
        spec = {"type": "exact_match", "path": "severity"}
        assert score(spec, '{"severity": "SEV2"}', expected="SEV2").passed

    def test_path_missing_in_output(self) -> None:
        spec = {"type": "exact_match", "path": "severity"}
        result = score(spec, '{"other": 1}', expected={"severity": "SEV2"})
        assert not result.passed
        assert result.detail is not None
        assert "not found" in result.detail

    def test_invalid_json_output_fails_with_detail(self) -> None:
        result = score({"type": "exact_match", "path": "a"}, "oops", expected={"a": 1})
        assert not result.passed
        assert result.detail is not None
        assert result.detail.startswith("output is not valid JSON")

    def test_requires_expected(self) -> None:
        scorer = build_scorer(SPEC.validate_python({"type": "exact_match"}), Path())
        assert scorer.check_case(make_case()) == "scorer 'exact_match' needs an 'expected' value"
        assert scorer.check_case(make_case(expected="x")) is None


class TestRegex:
    def test_match(self) -> None:
        assert score({"type": "regex", "pattern": r"SEV[1-4]"}, "Severity: SEV2").passed

    def test_no_match(self) -> None:
        result = score({"type": "regex", "pattern": r"^SEV"}, "Severity: SEV2")
        assert not result.passed
        assert result.detail == "did not match /^SEV/"

    def test_negate_fails_on_match(self) -> None:
        spec = {"type": "regex", "pattern": r"(?i)password", "negate": True}
        result = score(spec, "the Password is hunter2")
        assert not result.passed
        assert result.scorer == "not_regex"

    def test_ignore_case(self) -> None:
        assert score({"type": "regex", "pattern": "sev1", "ignore_case": True}, "SEV1").passed

    def test_path_into_json(self) -> None:
        spec = {"type": "regex", "pattern": r"^\d{4}-\d{2}-\d{2}", "path": "started_at"}
        assert score(spec, '{"started_at": "2026-01-02T03:04:05Z"}').passed

    def test_invalid_pattern_is_rejected_at_load(self) -> None:
        with pytest.raises(ValidationError, match="invalid regex"):
            SPEC.validate_python({"type": "regex", "pattern": "("})


class TestJsonSchema:
    SCHEMA: ClassVar[dict[str, object]] = {
        "type": "object",
        "required": ["severity", "summary"],
        "properties": {
            "severity": {"enum": ["SEV1", "SEV2", "SEV3"]},
            "summary": {"type": "string", "maxLength": 20},
        },
        "additionalProperties": False,
    }

    def test_valid(self) -> None:
        spec = {"type": "json_schema", "schema": self.SCHEMA}
        assert score(spec, '{"severity": "SEV1", "summary": "db down"}').passed

    def test_invalid_reports_paths(self) -> None:
        spec = {"type": "json_schema", "schema": self.SCHEMA}
        result = score(spec, '{"severity": "SEV9"}')
        assert not result.passed
        assert result.detail is not None
        assert "'summary' is a required property" in result.detail
        assert "$.severity" in result.detail

    def test_truncates_long_error_lists(self) -> None:
        schema = {"type": "array", "items": {"type": "integer"}}
        result = score({"type": "json_schema", "schema": schema}, '["a","b","c","d","e"]')
        assert result.detail is not None
        assert result.detail.endswith("… and 2 more")

    def test_not_json(self) -> None:
        result = score({"type": "json_schema", "schema": self.SCHEMA}, "plain text")
        assert not result.passed

    def test_schema_from_file_relative_to_base_dir(self, tmp_path: Path) -> None:
        (tmp_path / "schema.json").write_text(json.dumps(self.SCHEMA))
        spec = {"type": "json_schema", "schema": "schema.json"}
        assert score(spec, '{"severity": "SEV3", "summary": "ok"}', base_dir=tmp_path).passed

    def test_missing_schema_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="cannot read JSON Schema"):
            build_scorer(
                SPEC.validate_python({"type": "json_schema", "schema": "nope.json"}), tmp_path
            )

    def test_invalid_schema(self) -> None:
        spec = SPEC.validate_python({"type": "json_schema", "schema": {"type": "strng"}})
        with pytest.raises(ConfigError, match="invalid JSON Schema"):
            build_scorer(spec, Path())


def test_unknown_scorer_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SPEC.validate_python({"type": "vibes"})


def test_custom_name_and_weight() -> None:
    scorer = build_scorer(
        SPEC.validate_python({"type": "regex", "pattern": "x", "name": "has-x", "weight": 3}),
        Path(),
    )
    assert (scorer.name, scorer.weight) == ("has-x", 3.0)
