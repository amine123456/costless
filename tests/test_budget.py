import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from costless.budget import check_budget
from costless.cli import app
from costless.config import BudgetSettings, PricingSettings, load_config
from costless.models import RunSummary
from costless.runner import run_suite
from tests.conftest import WriteFile

# Every call: 1000 input + 500 output tokens on a model priced at $1 / $2 per MTok
# => $0.001 + $0.001 = $0.002 per attempt.
APP = """
from costless.context import record_usage
from costless.models import Usage

def run(text):
    record_usage(Usage(provider="fake", model=text, input_tokens=1000, output_tokens=500))
    return text
"""


def project(write: WriteFile, module_name: str, *, budget: str = "", strict: bool = True) -> Path:
    write(f"{module_name}.py", APP)
    write(
        "cases.yaml",
        """
        - {id: a, input: priced-model, expected: priced-model}
        - {id: b, input: priced-model, expected: priced-model}
        - {id: c, input: priced-model, expected: priced-model}
        """,
    )
    return write(
        "costless.yaml",
        f"""
        version: 1
        target: {{type: python, callable: "{module_name}:run"}}
        datasets: [{{path: cases.yaml}}]
        run: {{repeats: 2, concurrency: 1}}
        scorers: [{{type: exact_match}}]
        pricing:
          strict: {str(strict).lower()}
          models:
            priced-*: {{input: 1, output: 2}}
        budget: {{{budget}}}
        """,
    )


def test_costs_are_attached_to_attempts_cases_and_run(write: WriteFile, module_name: str) -> None:
    result = asyncio.run(run_suite(load_config(project(write, module_name))))

    assert {a.cost_usd for a in result.attempts} == {0.002}
    assert {c.cost_mean_usd for c in result.cases} == {0.002}
    assert result.summary.cost_total_usd == pytest.approx(0.012)
    assert result.summary.cost_per_case_usd == pytest.approx(0.002)
    assert result.summary.unpriced_models == ()


def test_run_budget_stops_spending(write: WriteFile, module_name: str) -> None:
    config = project(write, module_name, budget="max_run_usd: 0.005")
    result = asyncio.run(run_suite(load_config(config)))

    # Concurrency 1: three attempts reach $0.006 >= $0.005, the other three are skipped.
    executed = [a for a in result.attempts if not a.skipped]
    assert len(executed) == 3
    assert result.summary.skipped_attempts == 3
    skipped = next(a for a in result.attempts if a.skipped)
    assert skipped.error == "skipped: run budget of $0.005 reached"
    assert not skipped.passed
    assert result.summary.cost_total_usd == pytest.approx(0.006)


def summary(**overrides: object) -> RunSummary:
    fields: dict[str, object] = {
        "cases": 1,
        "attempts": 1,
        "quality_mean": 1.0,
        "failure_rate": 0.0,
        "error_rate": 0.0,
        "latency_p50_ms": 1.0,
        "latency_p95_ms": 1.0,
        "input_tokens": 1,
        "output_tokens": 1,
        "flaky_cases": 0,
        "cost_total_usd": 1.0,
        "cost_per_case_usd": 0.01,
    }
    fields.update(overrides)
    return RunSummary.model_validate(fields)


def test_check_budget_rules() -> None:
    strict = PricingSettings()
    assert check_budget(summary(), BudgetSettings(), strict) == []
    assert check_budget(summary(), BudgetSettings(max_run_usd=2, max_case_usd=0.02), strict) == []

    over = check_budget(summary(), BudgetSettings(max_run_usd=0.5, max_case_usd=0.001), strict)
    assert [v.rule for v in over] == ["max_run_usd", "max_case_usd"]
    assert over[0].message == "run cost $1.0000 exceeds $0.5"

    unpriced = summary(unpriced_models=("grok-4",), cost_total_usd=None, cost_per_case_usd=None)
    assert [v.rule for v in check_budget(unpriced, BudgetSettings(), strict)] == ["pricing"]
    assert check_budget(unpriced, BudgetSettings(), PricingSettings(strict=False)) == []


def test_cli_exit_codes(write: WriteFile, module_name: str, tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "run.json"

    ok = runner.invoke(app, ["run", "-c", str(project(write, module_name)), "-o", str(out)])
    assert ok.exit_code == 0, ok.output
    assert "cost total         $0.012000" in ok.stdout
    assert "cost per case      $0.002000" in ok.stdout

    over = runner.invoke(
        app,
        [
            "run",
            "-c",
            str(project(write, module_name, budget="max_case_usd: 0.001")),
            "-o",
            str(out),
        ],
    )
    assert over.exit_code == 1
    assert "BUDGET GATE FAILED [max_case_usd]" in over.stderr
    assert json.loads(out.read_text())["summary"]["cost_per_case_usd"] == 0.002  # still written


def test_unpriced_models_fail_strict_runs(
    write: WriteFile, module_name: str, tmp_path: Path
) -> None:
    config = project(write, module_name)
    write(
        "cases.yaml",
        "- {id: a, input: mystery-model, expected: mystery-model}\n",
    )
    result = CliRunner().invoke(app, ["run", "-c", str(config), "-o", str(tmp_path / "r.json")])
    assert result.exit_code == 1
    assert "no price for model(s) mystery-model" in result.stderr
    assert "unpriced models    mystery-model" in result.stdout
