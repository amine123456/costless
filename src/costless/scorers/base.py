"""Scorer interface."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from costless.models import Case, ScoreResult


@runtime_checkable
class Scorer(Protocol):
    """Anything that can grade one output of the system under test."""

    @property
    def name(self) -> str: ...

    @property
    def weight(self) -> float: ...

    def check_case(self, case: Case) -> str | None:
        """Return a problem description if this scorer cannot grade ``case``, else None.

        Called before a run starts, so dataset mistakes fail fast instead of
        silently showing up as quality regressions.
        """
        ...

    async def score(self, case: Case, output: str) -> ScoreResult: ...


class DeterministicScorer(ABC):
    """Base for scorers that are pure functions of (case, output)."""

    requires_expected: bool = False

    def __init__(self, name: str, weight: float) -> None:
        self._name = name
        self._weight = weight

    @property
    def name(self) -> str:
        return self._name

    @property
    def weight(self) -> float:
        return self._weight

    def check_case(self, case: Case) -> str | None:
        if self.requires_expected and case.expected is None:
            return f"scorer {self.name!r} needs an 'expected' value"
        return None

    async def score(self, case: Case, output: str) -> ScoreResult:
        return self.evaluate(case, output)

    @abstractmethod
    def evaluate(self, case: Case, output: str) -> ScoreResult: ...

    def _result(self, *, passed: bool, detail: str | None = None) -> ScoreResult:
        return ScoreResult(
            scorer=self.name, score=1.0 if passed else 0.0, passed=passed, detail=detail
        )
