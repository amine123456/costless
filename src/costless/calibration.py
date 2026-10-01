"""Judge calibration: how often does the LLM judge agree with humans?

A small set of outputs is scored by people (``human_score``) and by the judge.
The report gives, with 95% Wilson intervals where it is a proportion:

- **pass agreement**: judge and human give the same pass/fail verdict. This is
  what decides whether a case passes, so it is the headline number.
- **exact agreement** and **within-one agreement** on the raw scale.
- **Cohen's kappa** on pass/fail and **quadratic-weighted kappa** on the scale:
  agreement corrected for what chance alone would produce.
- **bias**: mean(judge - human). Positive means the judge is more lenient.
- **self-consistency**: share of examples where every repeat gave the same score.

With 20-30 examples the intervals are wide; that is the honest answer, and the
reason the intervals are reported at all.
"""

import asyncio
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from costless.context import attempt_scope
from costless.errors import ConfigError, DatasetError
from costless.metrics import cohens_kappa, mean, weighted_kappa, wilson_interval
from costless.models import CASE_ID_PATTERN, Case, Usage
from costless.pricing import PricingTable, to_report
from costless.scorers.llm_judge import JudgeError, LLMJudgeScorer


class LabeledExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=CASE_ID_PATTERN)
    input: Any = None
    output: str
    expected: Any = None
    rubric: str | None = None
    human_score: int
    notes: str | None = None


class ExampleResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    human_score: int
    judge_scores: tuple[int, ...]
    judge_score: int | None = Field(description="Median of the repeats (lower median).")
    errors: int
    reasoning: str | None
    pass_agreement: bool | None


class Proportion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float
    successes: int
    n: int
    ci95: tuple[float, float]


class CalibrationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    judge: str
    model: str
    labels_file: str
    scale: tuple[int, int]
    pass_threshold: int
    repeats: int
    examples: int
    scored_examples: int
    judge_errors: int
    pass_agreement: Proportion
    exact_agreement: Proportion
    within_one_agreement: Proportion
    pass_kappa: float | None
    weighted_kappa: float | None
    bias: float
    mean_abs_error: float
    self_consistency: float
    cost_usd: float | None
    unpriced_models: tuple[str, ...]
    results: tuple[ExampleResult, ...]

    @property
    def disagreements(self) -> list[ExampleResult]:
        return [r for r in self.results if r.pass_agreement is False]


def load_labels(path: Path) -> list[LabeledExample]:
    if not path.is_file():
        msg = f"labels file not found: {path}"
        raise DatasetError(msg)
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".jsonl":
            records: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            doc = yaml.safe_load(text)
            records = doc.get("examples") if isinstance(doc, dict) else doc
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        msg = f"{path}: cannot parse labels: {exc}"
        raise DatasetError(msg) from exc
    if not isinstance(records, list) or not records:
        msg = f"{path}: expected a non-empty list of examples (or an 'examples' list)"
        raise DatasetError(msg)
    try:
        examples = [LabeledExample.model_validate(r) for r in records]
    except ValidationError as exc:
        msg = f"{path}: {exc}"
        raise DatasetError(msg) from exc
    ids = [e.id for e in examples]
    if len(ids) != len(set(ids)):
        msg = f"{path}: duplicate example ids"
        raise DatasetError(msg)
    return examples


def check_labels(judge: LLMJudgeScorer, examples: Sequence[LabeledExample]) -> None:
    spec = judge.spec
    problems = [
        f"{e.id}: human_score {e.human_score} outside {spec.scale_min}..{spec.scale_max}"
        for e in examples
        if not spec.scale_min <= e.human_score <= spec.scale_max
    ]
    problems += [
        f"{e.id}: no rubric (neither the judge nor the example has one)"
        for e in examples
        if not (spec.rubric or e.rubric)
    ]
    if problems:
        raise ConfigError("invalid calibration labels:\n  " + "\n  ".join(problems))


async def calibrate(
    judge: LLMJudgeScorer,
    examples: Sequence[LabeledExample],
    *,
    labels_file: str,
    pricing: PricingTable,
    repeats: int = 3,
    concurrency: int = 4,
) -> CalibrationReport:
    check_labels(judge, examples)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(
        example: LabeledExample, repeat: int
    ) -> tuple[int | None, str | None, list[Usage]]:
        case = Case(
            id=example.id, input=example.input, expected=example.expected, rubric=example.rubric
        )
        async with semaphore:
            with attempt_scope(f"calibration/{example.id}", repeat) as scope:
                try:
                    score, reasoning = await judge.judge(case, example.output)
                except JudgeError:
                    return None, None, scope.usage
            return score, reasoning, scope.usage

    outcomes = await asyncio.gather(*(one(e, r) for e in examples for r in range(repeats)))
    usage: list[Usage] = [u for _, _, used in outcomes for u in used]

    threshold = judge.spec.threshold
    results: list[ExampleResult] = []
    for i, example in enumerate(examples):
        mine = outcomes[i * repeats : (i + 1) * repeats]
        scores = tuple(s for s, _, _ in mine if s is not None)
        reasoning = next((r for _, r, _ in mine if r), None)
        judge_score = int(statistics.median_low(scores)) if scores else None
        results.append(
            ExampleResult(
                id=example.id,
                human_score=example.human_score,
                judge_scores=scores,
                judge_score=judge_score,
                errors=repeats - len(scores),
                reasoning=reasoning,
                pass_agreement=None
                if judge_score is None
                else (judge_score >= threshold) == (example.human_score >= threshold),
            )
        )

    scored = [r for r in results if r.judge_score is not None]
    human = [r.human_score for r in scored]
    judged = [r.judge_score for r in scored if r.judge_score is not None]
    spec = judge.spec
    cost = pricing.cost(usage)
    return CalibrationReport(
        judge=judge.name,
        model=judge.model,
        labels_file=labels_file,
        scale=(spec.scale_min, spec.scale_max),
        pass_threshold=threshold,
        repeats=repeats,
        examples=len(examples),
        scored_examples=len(scored),
        judge_errors=sum(r.errors for r in results),
        pass_agreement=_proportion([r.pass_agreement is True for r in scored]),
        exact_agreement=_proportion([h == j for h, j in zip(human, judged, strict=True)]),
        within_one_agreement=_proportion(
            [abs(h - j) <= 1 for h, j in zip(human, judged, strict=True)]
        ),
        pass_kappa=cohens_kappa(
            [int(h >= threshold) for h in human], [int(j >= threshold) for j in judged]
        ),
        weighted_kappa=weighted_kappa(
            human, judged, categories=range(spec.scale_min, spec.scale_max + 1)
        ),
        bias=mean([j - h for h, j in zip(human, judged, strict=True)]),
        mean_abs_error=mean([abs(j - h) for h, j in zip(human, judged, strict=True)]),
        self_consistency=mean([float(len(set(r.judge_scores)) == 1) for r in scored]),
        cost_usd=to_report(cost.usd) if cost.complete else None,
        unpriced_models=tuple(sorted(cost.unpriced_models)),
        results=tuple(results),
    )


def check_calibration(
    report: CalibrationReport, *, min_pass_agreement: float | None, min_kappa: float | None
) -> list[str]:
    """Return why the judge is not trustworthy enough to gate on (empty if it is)."""
    failures: list[str] = []
    if report.scored_examples == 0:
        return ["the judge produced no usable verdict"]
    if min_pass_agreement is not None and report.pass_agreement.value < min_pass_agreement:
        failures.append(
            f"pass agreement {report.pass_agreement.value:.0%} < required {min_pass_agreement:.0%}"
        )
    if min_kappa is not None:
        if report.weighted_kappa is None:
            failures.append("weighted kappa is undefined (the labels have no variety)")
        elif report.weighted_kappa < min_kappa:
            failures.append(
                f"weighted kappa {report.weighted_kappa:.2f} < required {min_kappa:.2f}"
            )
    return failures


def _proportion(flags: Sequence[bool]) -> Proportion:
    successes = sum(flags)
    n = len(flags)
    return Proportion(
        value=successes / n if n else 0.0,
        successes=successes,
        n=n,
        ci95=wilson_interval(successes, n),
    )
