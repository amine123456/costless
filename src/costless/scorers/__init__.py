"""Scorers grade one output of the system under test and return a score in [0, 1]."""

from pathlib import Path

from costless.scorers.base import DeterministicScorer, Scorer
from costless.scorers.exact import ExactMatchScorer
from costless.scorers.json_schema import JsonSchemaScorer
from costless.scorers.regex import RegexScorer
from costless.specs import ExactMatchSpec, JsonSchemaSpec, RegexSpec, ScorerSpec


def build_scorer(spec: ScorerSpec, base_dir: Path) -> Scorer:
    """Instantiate a scorer. ``base_dir`` resolves relative file references (schemas)."""
    match spec:
        case ExactMatchSpec():
            return ExactMatchScorer(spec)
        case RegexSpec():
            return RegexScorer(spec)
        case JsonSchemaSpec():
            return JsonSchemaScorer(spec, base_dir)


__all__ = ["DeterministicScorer", "Scorer", "build_scorer"]
