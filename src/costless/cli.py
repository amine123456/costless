"""Command-line entry point."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from costless import __version__
from costless.config import DEFAULT_CONFIG_PATH, load_config
from costless.errors import CostlessError
from costless.models import Attempt, RunResult
from costless.results import run_json_schema, write_run
from costless.runner import prepare, run_suite

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


@app.command()
def schema() -> None:
    """Print the JSON Schema of run.json."""
    typer.echo(json.dumps(run_json_schema(), indent=2))


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
        ("flaky cases", str(s.flaky_cases)),
    ]
    width = max(len(label) for label, _ in rows)
    lines = [f"{label.ljust(width)}  {value}" for label, value in rows]
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


def _fail(exc: CostlessError) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(EXIT_CONFIG_ERROR)
