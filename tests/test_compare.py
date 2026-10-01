import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from costless.cli import app
from costless.compare import Comparison, MetricComparison, Status, compare_runs
from costless.config import GateSettings, QualityGate
from costless.report import MARKER, render_markdown
from costless.results import write_run
from tests.builders import make_run

GATE = GateSettings(bootstrap_samples=2000)


def stable(n_cases: int = 20, value: float = 1.0, repeats: int = 5) -> dict[str, list[float]]:
    return {f"c{i:02d}": [value] * repeats for i in range(n_cases)}


def metric(comparison: Comparison, name: str) -> MetricComparison:
    return next(m for m in comparison.metrics if m.metric == name)


def test_unchanged_runs_pass() -> None:
    base = make_run(stable(), ref="main", sha="a" * 40)
    cand = make_run(stable())
    result = compare_runs(cand, base, GATE)

    assert result.gate_passed
    assert result.paired_cases == 20
    assert {m.status for m in result.metrics} == {Status.OK}
    assert result.regressed_cases == ()


def test_clear_quality_regression_blocks_and_explains() -> None:
    base_q = stable()
    cand_q = dict(base_q)
    for i in range(8):  # 8 of 20 cases now fail every time
        cand_q[f"c{i:02d}"] = [0.0] * 5
    result = compare_runs(make_run(cand_q), make_run(base_q, ref="main"), GATE)

    quality = metric(result, "quality")
    assert quality.status == Status.REGRESSED
    assert quality.change == pytest.approx(-0.4)
    assert quality.ci is not None
    assert quality.ci[1] < 0
    assert metric(result, "failure_rate").status == Status.REGRESSED
    assert not result.gate_passed
    assert result.reasons[0].startswith("quality: regressed")
    assert len(result.regressed_cases) == 8
    assert result.regressed_cases[0].example_failure == "judge: c00 | wrong answer"


def test_drop_within_tolerance_is_ok() -> None:
    base_q = stable(value=0.9)
    cand_q = stable(value=0.89)  # -0.01 < tolerance 0.02, but perfectly consistent
    result = compare_runs(make_run(cand_q), make_run(base_q), GATE)
    assert metric(result, "quality").status == Status.OK
    assert result.gate_passed


def test_noisy_drop_is_inconclusive_not_blocking() -> None:
    # Only one of four cases got worse: the mean drops by 1/12 (past the 0.02
    # tolerance), but resamples that leave that case out show no change, so the
    # interval reaches 0 and the evidence is not conclusive.
    base_q = {f"c{i}": [1.0, 1.0, 1.0] for i in range(4)}
    cand_q = {**base_q, "c0": [0.0, 1.0, 1.0]}
    result = compare_runs(make_run(cand_q), make_run(base_q), GATE)
    quality = metric(result, "quality")
    assert quality.status == Status.INCONCLUSIVE
    assert quality.ci is not None
    assert quality.ci[1] == 0
    assert result.gate_passed

    strict = compare_runs(
        make_run(cand_q), make_run(base_q), GATE.model_copy(update={"fail_on_inconclusive": True})
    )
    assert not strict.gate_passed


def test_improvement() -> None:
    result = compare_runs(make_run(stable(value=1.0)), make_run(stable(value=0.0)), GATE)
    assert metric(result, "quality").status == Status.IMPROVED
    assert result.gate_passed
    assert len(result.fixed_cases) == 15  # capped list


def test_cost_and_latency_ratios() -> None:
    base = make_run(
        stable(), cost_usd=0.001, latency_ms={k: [100, 110, 120, 130, 140] for k in stable()}
    )
    cand = make_run(
        stable(), cost_usd=0.0015, latency_ms={k: [200, 220, 240, 260, 280] for k in stable()}
    )
    result = compare_runs(cand, base, GATE)
    cost = metric(result, "cost_per_case")
    assert cost.status == Status.REGRESSED
    assert cost.change == pytest.approx(1.5)
    latency = metric(result, "latency_p95_ms")
    assert latency.status == Status.REGRESSED
    assert latency.change == pytest.approx(2.0)


def test_unpriced_cost_is_unavailable_not_blocking() -> None:
    result = compare_runs(make_run(stable(), cost_usd=None), make_run(stable()), GATE)
    assert metric(result, "cost_per_case").status == Status.UNAVAILABLE
    assert result.gate_passed


def test_absolute_limits_without_baseline() -> None:
    gate = GATE.model_copy(update={"quality": QualityGate(min=0.95)})
    result = compare_runs(make_run(stable(value=0.9)), None, gate)
    quality = metric(result, "quality")
    assert quality.status == Status.BREACHED
    assert quality.note == "0.9 violates the absolute limit 0.95"
    assert not result.gate_passed
    assert compare_runs(make_run(stable()), None, gate).gate_passed


def test_case_set_and_dataset_changes_are_reported() -> None:
    base_q = stable(4)
    cand_q = {**stable(3), "new": [1.0] * 5}
    result = compare_runs(make_run(cand_q, dataset_sha="1" * 64), make_run(base_q), GATE)
    assert result.paired_cases == 3
    assert result.only_in_candidate == ("new",)
    assert result.only_in_baseline == ("c03",)
    assert result.datasets_changed
    text = render_markdown(result)
    assert "New cases, not compared: `new`" in text
    assert "The datasets differ" in text


# ---------------------------------------------------------------- method properties


def _simulated_pair(
    rng: np.random.Generator, drop: float
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """30 cases with heterogeneous difficulty; each attempt passes with prob p_i."""
    p = rng.beta(4, 1.5, size=30)
    base = {f"c{i}": list((rng.random(5) < p[i]).astype(float)) for i in range(30)}
    p_cand = np.clip(p - drop, 0, 1)
    cand = {f"c{i}": list((rng.random(5) < p_cand[i]).astype(float)) for i in range(30)}
    return base, cand


# Reference, measured with a paired t-test on the same simulation (300 trials):
# 3% false alarms with no change, 85% detection of a 15-point drop. The bootstrap
# measured 4% and 87%. The bounds below leave room for Monte Carlo noise.


def test_false_alarm_rate_under_no_change() -> None:
    """With identical underlying behaviour, the gate should almost never block."""
    rng = np.random.default_rng(2024)
    gate = GateSettings(bootstrap_samples=1000)
    blocked = 0
    trials = 200
    for seed in range(trials):
        base, cand = _simulated_pair(rng, drop=0.0)
        result = compare_runs(
            make_run(cand), make_run(base), gate.model_copy(update={"seed": seed})
        )
        blocked += metric(result, "quality").status == Status.REGRESSED
    assert blocked / trials <= 0.05


def test_detects_a_real_drop_most_of_the_time() -> None:
    """A 15-point drop in pass probability is caught in most trials."""
    rng = np.random.default_rng(7)
    gate = GateSettings(bootstrap_samples=1000)
    caught = 0
    trials = 150
    for seed in range(trials):
        base, cand = _simulated_pair(rng, drop=0.15)
        result = compare_runs(
            make_run(cand), make_run(base), gate.model_copy(update={"seed": seed})
        )
        caught += metric(result, "quality").status == Status.REGRESSED
    assert caught / trials >= 0.75


# ---------------------------------------------------------------- report & CLI


def test_markdown_report() -> None:
    base_q = stable()
    broken = {f"c{i:02d}": [0.0] * 5 for i in range(5)}
    cand_q = {**base_q, **broken}
    text = render_markdown(
        compare_runs(
            make_run(cand_q, ref="feat/prompt", sha="b" * 40),
            make_run(base_q, ref="main", sha="a" * 40),
            GATE,
        )
    )
    assert text.startswith(MARKER)
    assert "## ❌ costless: quality gate failed" in text
    assert (
        "Compared `feat/prompt` (bbbbbbbb) with `main` (aaaaaaaa): 20 cases, 5 repeats each" in text
    )
    assert "| Quality (mean score) | 1.000 | 0.750 | -0.250 |" in text
    assert "❌ regressed" in text
    assert "### What got worse (5 cases)" in text
    assert "| `c00` | 1.00 → 0.00 | 100% → 0% | judge: c00 \\| wrong answer |" in text


def test_markdown_without_baseline() -> None:
    text = render_markdown(compare_runs(make_run(stable()), None, GATE))
    assert "No baseline run available" in text
    assert "✅ costless: quality gate passed" in text


def test_cli_compare(tmp_path: Path) -> None:
    base, cand = tmp_path / "base.json", tmp_path / "cand.json"
    write_run(make_run(stable(), ref="main"), base)
    write_run(make_run({**stable(), **{f"c{i:02d}": [0.0] * 5 for i in range(5)}}), cand)
    md, js = tmp_path / "report.md", tmp_path / "comparison.json"
    runner = CliRunner()

    failed = runner.invoke(
        app,
        [
            "compare",
            "--candidate",
            str(cand),
            "-b",
            str(base),
            "--markdown",
            str(md),
            "--json",
            str(js),
        ],
    )
    assert failed.exit_code == 1, failed.output
    assert md.read_text().startswith(MARKER)
    assert json.loads(js.read_text())["gate_passed"] is False

    ok = runner.invoke(app, ["compare", "--candidate", str(base), "-b", str(base)])
    assert ok.exit_code == 0

    missing = runner.invoke(
        app, ["compare", "--candidate", str(base), "-b", str(tmp_path / "nope.json")]
    )
    assert missing.exit_code == 2
    assert "baseline not found" in missing.stderr

    first_run = runner.invoke(
        app,
        [
            "compare",
            "--candidate",
            str(base),
            "-b",
            str(tmp_path / "nope.json"),
            "--allow-missing-baseline",
        ],
    )
    assert first_run.exit_code == 0
    assert "absolute limits only" in first_run.stderr
