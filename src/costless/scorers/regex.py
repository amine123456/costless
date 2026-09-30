import re

from costless.models import Case, ScoreResult
from costless.scorers._json import PathNotFoundError, as_text, get_path, parse_json_output
from costless.scorers.base import DeterministicScorer
from costless.specs import RegexSpec


class RegexScorer(DeterministicScorer):
    """Search the output (or one field of the JSON output) for a pattern."""

    def __init__(self, spec: RegexSpec) -> None:
        super().__init__(spec.name or ("not_regex" if spec.negate else "regex"), spec.weight)
        self._spec = spec
        self._pattern = re.compile(spec.pattern, re.IGNORECASE if spec.ignore_case else 0)

    def evaluate(self, case: Case, output: str) -> ScoreResult:
        text = output
        if self._spec.path is not None:
            try:
                text = as_text(get_path(parse_json_output(output), self._spec.path))
            except ValueError as exc:
                return self._result(passed=False, detail=f"output is not valid JSON: {exc}")
            except PathNotFoundError as exc:
                return self._result(passed=False, detail=f"output: {exc}")

        found = self._pattern.search(text) is not None
        if found != self._spec.negate:
            return self._result(passed=True)
        verb = "matched" if self._spec.negate else "did not match"
        return self._result(passed=False, detail=f"{verb} /{self._spec.pattern}/")
