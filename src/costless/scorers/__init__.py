"""Scorers grade one output of the system under test and return a score in [0, 1]."""

from pathlib import Path

from costless.scorers.base import DeterministicScorer, Scorer
from costless.scorers.exact import ExactMatchScorer
from costless.scorers.json_schema import JsonSchemaScorer
from costless.scorers.llm_judge import LLMJudgeScorer
from costless.scorers.regex import RegexScorer
from costless.specs import ExactMatchSpec, JsonSchemaSpec, LLMJudgeSpec, RegexSpec, ScorerSpec


def build_scorer(spec: ScorerSpec, base_dir: Path) -> Scorer:
    """Instantiate a scorer. ``base_dir`` resolves relative file references (schemas)."""
    match spec:
        case ExactMatchSpec():
            return ExactMatchScorer(spec)
        case RegexSpec():
            return RegexScorer(spec)
        case JsonSchemaSpec():
            return JsonSchemaScorer(spec, base_dir)
        case LLMJudgeSpec():
            return LLMJudgeScorer.from_spec(spec)


__all__ = ["DeterministicScorer", "LLMJudgeScorer", "Scorer", "build_scorer"]
