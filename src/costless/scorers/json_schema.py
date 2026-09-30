import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from costless.errors import ConfigError
from costless.models import Case, ScoreResult
from costless.scorers._json import parse_json_output
from costless.scorers.base import DeterministicScorer
from costless.specs import JsonSchemaSpec

_MAX_REPORTED_ERRORS = 3


class JsonSchemaScorer(DeterministicScorer):
    """The output must be valid JSON conforming to a JSON Schema (draft 2020-12)."""

    def __init__(self, spec: JsonSchemaSpec, base_dir: Path) -> None:
        super().__init__(spec.name or "json_schema", spec.weight)
        schema = _load_schema(spec.json_schema, base_dir)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            msg = f"invalid JSON Schema: {exc.message}"
            raise ConfigError(msg) from exc
        self._validator = Draft202012Validator(schema)

    def evaluate(self, case: Case, output: str) -> ScoreResult:
        try:
            document = parse_json_output(output)
        except ValueError as exc:
            return self._result(passed=False, detail=f"output is not valid JSON: {exc}")

        errors = sorted(self._validator.iter_errors(document), key=lambda e: list(e.absolute_path))
        if not errors:
            return self._result(passed=True)
        messages = [f"{e.json_path}: {e.message}" for e in errors[:_MAX_REPORTED_ERRORS]]
        if len(errors) > _MAX_REPORTED_ERRORS:
            messages.append(f"… and {len(errors) - _MAX_REPORTED_ERRORS} more")
        return self._result(passed=False, detail="; ".join(messages))


def _load_schema(source: dict[str, Any] | str, base_dir: Path) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = Path(source)
    if not path.is_absolute():
        path = base_dir / path
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"cannot read JSON Schema {path}: {exc.strerror}"
        raise ConfigError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"{path}: invalid JSON: {exc.msg}"
        raise ConfigError(msg) from exc
    if not isinstance(loaded, dict):
        msg = f"{path}: a JSON Schema must be an object"
        raise ConfigError(msg)
    return loaded
