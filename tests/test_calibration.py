import asyncio
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from costless.calibration import (
    CalibrationReport,
    calibrate,
    check_calibration,
    load_labels,
)
from costless.cli import app
from costless.errors import ConfigError, DatasetError
from costless.pricing import PricingTable
from costless.scorers.llm_judge import LLMJudgeScorer
from costless.specs import LLMJudgeSpec
from tests.conftest import WriteFile
from tests.fakes import ScriptedProvider, verdict

# human score, and what the judge says for each repeat
SCENARIO = {
    "agree-high": (5, [5, 5, 5]),
    "agree-low": (1, [1, 1, 1]),
    "off-by-one": (4, [5, 5, 4]),  # same pass verdict, not exact
    "lenient": (2, [4, 4, 4]),  # pass/fail disagreement
    "unstable": (3, [2, 3, 2]),
}

LABELS = "\n".join(
    f"- {{id: {key}, input: 'report {key}', output: 'summary {key}', human_score: {human}}}"
    for key, (human, _) in SCENARIO.items()
)


def scripted_judge() -> tuple[LLMJudgeScorer, ScriptedProvider]:
    calls: dict[str, int] = {}

    def reply(prompt: str) -> str:
        key = re.search(r"<response>\nsummary (\S+)\n", prompt).group(1)  # type: ignore[union-attr]
        n = calls.get(key, 0)
        calls[key] = n + 1
        return verdict(SCENARIO[key][1][n % 3])

    provider = ScriptedProvider(reply, model="claude-haiku-4-5")
    spec = LLMJudgeSpec.model_validate({"type": "llm_judge", "rubric": "Accurate summary."})
    return LLMJudgeScorer(spec, provider, "claude-haiku-4-5"), provider


def run_calibration(write: WriteFile) -> CalibrationReport:
    examples = load_labels(write("labels.yaml", LABELS))
    scorer, _ = scripted_judge()
    return asyncio.run(
        calibrate(
            scorer,
            examples,
            labels_file="labels.yaml",
            pricing=PricingTable.default(),
            repeats=3,
            concurrency=1,
        )
    )


def test_report_metrics(write: WriteFile) -> None:
    report = run_calibration(write)
    by_id = {r.id: r for r in report.results}

    assert report.examples == report.scored_examples == 5
    assert by_id["off-by-one"].judge_score == 5  # lower median of 5, 5, 4
    assert by_id["unstable"].judge_score == 2

    # pass threshold 4: verdicts agree everywhere except "lenient"
    assert report.pass_agreement.successes == 4
    assert report.pass_agreement.n == 5
    assert report.pass_agreement.ci95[0] < 0.8 < report.pass_agreement.ci95[1]
    assert [r.id for r in report.disagreements] == ["lenient"]

    assert report.exact_agreement.successes == 2  # agree-high, agree-low
    assert report.within_one_agreement.successes == 4  # all but lenient
    assert report.bias == pytest.approx((0 + 0 + 1 + 2 - 1) / 5)
    assert report.mean_abs_error == pytest.approx((0 + 0 + 1 + 2 + 1) / 5)
    assert report.self_consistency == pytest.approx(3 / 5)
    assert report.pass_kappa is not None
    assert 0 < report.pass_kappa < 1
    assert report.weighted_kappa is not None
    # 15 judge calls of 100 input + 20 output tokens on Haiku 4.5 ($1 / $5 per MTok)
    assert report.cost_usd == pytest.approx(15 * (100 * 1 + 20 * 5) / 1e6)


def test_check_calibration(write: WriteFile) -> None:
    report = run_calibration(write)
    assert check_calibration(report, min_pass_agreement=0.8, min_kappa=None) == []
    failures = check_calibration(report, min_pass_agreement=0.9, min_kappa=0.99)
    assert failures[0] == "pass agreement 80% < required 90%"
    assert failures[1].startswith("weighted kappa")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[]\n", "non-empty list"),
        ("- {id: a, output: x}\n", "human_score"),
        (
            "- {id: a, output: x, human_score: 1}\n- {id: a, output: y, human_score: 2}\n",
            "duplicate",
        ),
        ("examples: [\n", "cannot parse"),
    ],
)
def test_invalid_label_files(write: WriteFile, content: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        load_labels(write("labels.yaml", content))


def test_jsonl_labels(write: WriteFile) -> None:
    path = write("labels.jsonl", json.dumps({"id": "a", "output": "x", "human_score": 3}) + "\n")
    assert load_labels(path)[0].human_score == 3


def test_labels_outside_scale_are_rejected(write: WriteFile) -> None:
    scorer, _ = scripted_judge()
    examples = load_labels(write("labels.yaml", "- {id: a, output: x, human_score: 7}\n"))
    with pytest.raises(ConfigError, match=r"human_score 7 outside 1\.\.5"):
        asyncio.run(calibrate(scorer, examples, labels_file="l", pricing=PricingTable.default()))


def test_cli(write: WriteFile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scorer, _ = scripted_judge()
    monkeypatch.setattr(LLMJudgeScorer, "from_spec", classmethod(lambda cls, spec: scorer))
    write("labels.yaml", LABELS)
    write("cases.yaml", "- {id: a, input: x, rubric: r}\n")

    def config(thresholds: str) -> str:
        return str(
            write(
                "costless.yaml",
                f"""
                version: 1
                target: {{type: python, callable: "nothing:run"}}
                datasets: [{{path: cases.yaml}}]
                scorers:
                  - type: llm_judge
                    name: accuracy
                    rubric: Accurate summary.
                    calibration: {{labels: labels.yaml, {thresholds}}}
                """,
            )
        )

    out = tmp_path / "cal.json"
    runner = CliRunner()
    ok = runner.invoke(app, ["calibrate", "-c", config("min_pass_agreement: 0.8"), "-o", str(out)])
    assert ok.exit_code == 0, ok.output
    assert re.search(r"pass agreement\s+80% \(4/5, 95% CI \d+%-\d+%\)", ok.stdout)
    assert "lenient: human 2, judge 4 [4, 4, 4]" in ok.stdout
    assert json.loads(out.read_text())["pass_agreement"]["successes"] == 4

    strict = runner.invoke(
        app, ["calibrate", "-c", config("min_pass_agreement: 0.95"), "-o", str(out)]
    )
    assert strict.exit_code == 1
    assert "CALIBRATION FAILED: pass agreement 80% < required 95%" in strict.stderr

    wrong_name = runner.invoke(app, ["calibrate", "-c", config(""), "--judge", "tone"])
    assert wrong_name.exit_code == 2
    assert "no llm_judge scorer named 'tone'" in wrong_name.stderr
