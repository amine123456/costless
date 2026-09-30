import json
from typing import Any

from costless.models import Case, ScoreResult
from costless.scorers._json import PathNotFoundError, get_path, parse_json_output
from costless.scorers.base import DeterministicScorer
from costless.specs import ExactMatchSpec


class ExactMatchScorer(DeterministicScorer):
    """Compare the output, or one field of the JSON output, with ``case.expected``.

    - With ``path``: the field at ``path`` in the parsed output is compared with the
      field at the same path in ``expected`` (or with ``expected`` itself if it is a scalar).
    - Without ``path``: a string ``expected`` is compared with the raw output text; any
      other ``expected`` is compared with the parsed JSON output.

    String normalisation (whitespace, case) applies to string values only.
    """

    requires_expected = True

    def __init__(self, spec: ExactMatchSpec) -> None:
        super().__init__(
            spec.name or "exact_match" + (f"[{spec.path}]" if spec.path else ""), spec.weight
        )
        self._spec = spec

    def evaluate(self, case: Case, output: str) -> ScoreResult:
        try:
            actual, expected = self._operands(case, output)
        except ValueError as exc:
            return self._result(passed=False, detail=f"output is not valid JSON: {exc}")
        except PathNotFoundError as exc:
            return self._result(passed=False, detail=f"output: {exc}")

        if self._normalise(actual) == self._normalise(expected):
            return self._result(passed=True)
        return self._result(
            passed=False, detail=f"expected {_short(expected)}, got {_short(actual)}"
        )

    def _operands(self, case: Case, output: str) -> tuple[Any, Any]:
        path = self._spec.path
        if path is None:
            if isinstance(case.expected, str):
                return output, case.expected
            return parse_json_output(output), case.expected
        actual = get_path(parse_json_output(output), path)
        if isinstance(case.expected, dict | list):
            try:
                return actual, get_path(case.expected, path)
            except PathNotFoundError as exc:
                msg = f"case {case.id!r} expected value has no {path!r}"
                raise ValueError(msg) from exc
        return actual, case.expected

    def _normalise(self, value: Any) -> Any:  # noqa: ANN401
        if not isinstance(value, str):
            return value
        if self._spec.normalize_whitespace:
            value = " ".join(value.split())
        if not self._spec.case_sensitive:
            value = value.casefold()
        return value


def _short(value: Any, limit: int = 80) -> str:  # noqa: ANN401
    text = repr(value) if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 1] + "…"
