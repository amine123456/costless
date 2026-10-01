"""Command-line entry point."""

import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer

from costless import __version__
from costless.budget import check_budget
from costless.calibration import (
    CalibrationReport,
    Proportion,
    check_calibration,
    load_labels,
)
from costless.calibration import calibrate as calibrate_judge
from costless.compare import compare_runs
from costless.config import DEFAULT_CONFIG_PATH, GateSettings, load_config
from costless.errors import ConfigError, CostlessError
from costless.models import Attempt, RunResult
from costless.report import render_markdown
from costless.results import read_run, run_json_schema, write_run
from costless.runner import prepare, run_suite
from costless.scorers import LLMJudgeScorer
from costless.specs import LLMJudgeSpec, ScorerSpec

app = typer.Typer(
    name="costless",
    help="Evaluate an LLM application and gate merges on quality, cost and latency.",
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", help="Path to costless.yaml.", dir_okay=False)
]
TagOption = Annotated[
    list[str] | None, typer.Option("--tag", "-t", help="Only run cases with this tag (repeatable).")
]

EXIT_GATE_FAILED = 1
EXIT_CONFIG_ERROR = 2


@app.callback()
def main() -> None:
    """costless command-line interface."""


@app.command()
def version() -> None:
    """Print the installed costless version."""
    typer.echo(__version__)


@app.command()
def validate(config: ConfigOption = DEFAULT_CONFIG_PATH, tag: TagOption = None) -> None:
    """Check the config, datasets and scorers without calling the target."""
    try:
        loaded = load_config(config)
        datasets, items = prepare(loaded, tags=tag or ())
    except CostlessError as exc:
        _fail(exc)
    for dataset in datasets:
        version = f" v{dataset.version}" if dataset.version else ""
        typer.echo(
            f"dataset {dataset.name}{version}: {len(dataset.cases)} cases ({dataset.sha256[:12]})"
        )
    typer.echo(f"ok: {len(items)} cases ready to run")


@app.command()
def run(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write run.json.", dir_okay=False)
    ] = Path(".costless/run.json"),
    repeats: Annotated[
        int | None, typer.Option("--repeats", "-n", min=1, help="Override run.repeats.")
    ] = None,
    tag: TagOption = None,
) -> None:
    """Run every case N times against the target and write run.json."""
    try:
        loaded = load_config(config)
        result = asyncio.run(
            run_suite(loaded, repeats=repeats, tags=tag or (), on_attempt=_progress)
        )
    except CostlessError as exc:
        _fail(exc)
    typer.echo("", err=True)
    write_run(result, output)
    typer.echo(_render_summary(result))
    typer.echo(f"results written to {output}")

    violations = check_budget(result.summary, loaded.config.budget, loaded.config.pricing)
    if violations:
        typer.echo("")
        for violation in violations:
            typer.echo(f"BUDGET GATE FAILED [{violation.rule}]: {violation.message}", err=True)
        raise typer.Exit(EXIT_GATE_FAILED)


@app.command()
def compare(
    candidate: Annotated[
        Path, typer.Option("--candidate", help="run.json of this change.", dir_okay=False)
    ] = Path(".costless/run.json"),
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", "-b", help="run.json of the main branch.", dir_okay=False),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="costless.yaml with gate thresholds.", dir_okay=False),
    ] = None,
    allow_missing_baseline: Annotated[
        bool,
        typer.Option(help="If the baseline file does not exist, check absolute limits only."),
    ] = False,
    markdown: Annotated[
        Path | None, typer.Option("--markdown", help="Write the Markdown report here.")
    ] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the comparison as JSON here.")
    ] = None,
) -> None:
    """Compare this change with the baseline and fail on significant regressions."""
    try:
        gate = load_config(config).config.gate if config else GateSettings()
        candidate_run = read_run(candidate)
        baseline_run = None
        if baseline is not None:
            if baseline.exists():
                baseline_run = read_run(baseline)
            elif not allow_missing_baseline:
                msg = f"baseline not found: {baseline} (use --allow-missing-baseline on first runs)"
                raise ConfigError(msg)
            else:
                typer.echo(f"warning: no baseline at {baseline}; absolute limits only", err=True)
        comparison = compare_runs(candidate_run, baseline_run, gate)
    except CostlessError as exc:
        _fail(exc)

    report = render_markdown(comparison)
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(report, encoding="utf-8")
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(report)
    if not comparison.gate_passed:
        raise typer.Exit(EXIT_GATE_FAILED)


@app.command()
def calibrate(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    judge: Annotated[
        str | None, typer.Option("--judge", "-j", help="Name of the llm_judge scorer.")
    ] = None,
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels", "-l", help="Human labels; defaults to the judge's calibration.labels."
        ),
    ] = None,
    repeats: Annotated[
        int, typer.Option("--repeats", "-n", min=1, help="Judge runs per example.")
    ] = 3,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the report.", dir_okay=False)
    ] = Path(".costless/calibration.json"),
) -> None:
    """Measure how well an LLM judge agrees with human-labelled examples."""
    try:
        loaded = load_config(config)
        spec = _judge_spec(loaded.config.scorers, judge)
        labels_path = labels or (
            loaded.resolve(spec.calibration.labels) if spec.calibration else None
        )
        if labels_path is None:
            msg = "no labels: pass --labels or set calibration.labels on the judge"
            raise ConfigError(msg)
        examples = load_labels(labels_path)
        scorer = LLMJudgeScorer.from_spec(spec)
        report = asyncio.run(
            calibrate_judge(
                scorer,
                examples,
                labels_file=str(labels_path),
                pricing=loaded.pricing_table(),
                repeats=repeats,
                concurrency=loaded.config.run.concurrency,
            )
        )
    except CostlessError as exc:
        _fail(exc)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    typer.echo(_render_calibration(report))
    typer.echo(f"report written to {output}")

    thresholds = spec.calibration
    failures = check_calibration(
        report,
        min_pass_agreement=thresholds.min_pass_agreement if thresholds else None,
        min_kappa=thresholds.min_kappa if thresholds else None,
    )
    if failures:
        typer.echo("")
        for failure in failures:
            typer.echo(f"CALIBRATION FAILED: {failure}", err=True)
        raise typer.Exit(EXIT_GATE_FAILED)


@app.command()
def schema() -> None:
    """Print the JSON Schema of run.json."""
    typer.echo(json.dumps(run_json_schema(), indent=2))


def _judge_spec(scorers: Sequence[ScorerSpec], name: str | None) -> LLMJudgeSpec:
    judges = [s for s in scorers if isinstance(s, LLMJudgeSpec)]
    if name is not None:
        judges = [s for s in judges if (s.name or "llm_judge") == name]
    if not judges:
        msg = "no llm_judge scorer" + (f" named {name!r}" if name else "") + " in the config"
        raise ConfigError(msg)
    if len(judges) > 1:
        msg = "several llm_judge scorers in the config; choose one with --judge"
        raise ConfigError(msg)
    return judges[0]


def _render_calibration(report: CalibrationReport) -> str:
    def prop(p: Proportion) -> str:
        low, high = p.ci95
        return f"{p.value:.0%} ({p.successes}/{p.n}, 95% CI {low:.0%}-{high:.0%})"

    def kappa(value: float | None) -> str:
        return "undefined" if value is None else f"{value:.2f}"

    rows = [
        ("judge", f"{report.judge} ({report.model})"),
        (
            "examples",
            f"{report.scored_examples}/{report.examples} scored x {report.repeats} repeats",
        ),
        ("pass agreement", prop(report.pass_agreement)),
        ("exact agreement", prop(report.exact_agreement)),
        ("within one point", prop(report.within_one_agreement)),
        ("kappa (pass/fail)", kappa(report.pass_kappa)),
        ("kappa (weighted)", kappa(report.weighted_kappa)),
        ("bias (judge-human)", f"{report.bias:+.2f}"),
        ("self-consistency", f"{report.self_consistency:.0%}"),
        ("judge errors", str(report.judge_errors)),
        ("cost", _usd(report.cost_usd)),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"{label.ljust(width)}  {value}" for label, value in rows]
    if report.disagreements:
        lines += ["", "pass/fail disagreements:"]
        lines += [
            f"  {r.id}: human {r.human_score}, judge {r.judge_score} {list(r.judge_scores)}"
            + (f" - {r.reasoning[:100]}" if r.reasoning else "")
            for r in report.disagreements
        ]
    return "\n".join(lines)


def _progress(attempt: Attempt) -> None:
    mark = "E" if attempt.error else ("." if attempt.passed else "F")
    sys.stderr.write(mark)
    sys.stderr.flush()


def _render_summary(result: RunResult) -> str:
    s = result.summary
    rows = [
        ("cases x repeats", f"{s.cases} x {result.metadata.repeats} = {s.attempts} attempts"),
        ("quality (mean)", f"{s.quality_mean:.3f}"),
        ("failure rate", f"{s.failure_rate:.1%}"),
        ("error rate", f"{s.error_rate:.1%}"),
        ("latency p50 / p95", f"{s.latency_p50_ms:.0f} ms / {s.latency_p95_ms:.0f} ms"),
        ("tokens in / out", f"{s.input_tokens} / {s.output_tokens}"),
        ("cost total", _usd(s.cost_total_usd)),
        ("cost per case", _usd(s.cost_per_case_usd)),
        ("eval cost (judges)", _usd(s.eval_cost_total_usd)),
        ("flaky cases", str(s.flaky_cases)),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"{label.ljust(width)}  {value}" for label, value in rows]
    if s.unpriced_models:
        lines.append(f"{'unpriced models'.ljust(width)}  {', '.join(s.unpriced_models)}")
    if s.scorer_errors:
        lines.append(
            f"{'scorer errors'.ljust(width)}  {s.scorer_errors} (not verdicts; see run.json)"
        )
    if s.skipped_attempts:
        lines.append(f"{'skipped (budget)'.ljust(width)}  {s.skipped_attempts} attempts")
    failing = [c for c in result.cases if c.pass_rate < 1]
    if failing:
        lines.append("")
        lines.append("cases not passing on every repeat:")
        lines.extend(
            f"  {c.case_id}: pass {c.pass_rate:.0%}, quality {c.quality_mean:.2f}"
            + (f", {c.errors} errors" if c.errors else "")
            for c in failing
        )
    return "\n".join(lines)


def _usd(value: float | None) -> str:
    return "unpriced" if value is None else f"${value:.6f}"


def _fail(exc: CostlessError) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(EXIT_CONFIG_ERROR)
