import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from costless.config import load_config
from costless.errors import ConfigError
from costless.models import RunResult
from costless.results import read_run, write_run
from costless.runner import run_suite
from tests.conftest import WriteFile

# A tiny system under test. It fails deterministically on some inputs, crashes
# on others, and is non-deterministic on "coin" inputs (passes on even repeats),
# so a single run exercises every outcome the runner has to aggregate.
APP = """
import asyncio, json
from costless.context import current_attempt, record_usage
from costless.models import Usage

async def run(text):
    record_usage(Usage(provider="fake", model="m", input_tokens=len(text), output_tokens=3))
    if text == "crash":
        raise RuntimeError("boom")
    if text == "slow":
        await asyncio.sleep(5)
    if text == "coin":
        severity = "SEV1" if current_attempt().repeat % 2 == 0 else "SEV3"
        return json.dumps({"severity": severity})
    return json.dumps({"severity": "SEV1" if "down" in text else "SEV3"})

def sync_run(text):
    return {"severity": "SEV1"}
"""

CASES = """
dataset: incidents
version: "1"
cases:
  - {id: down, input: "db down", expected: {severity: SEV1}, tags: [db]}
  - {id: wrong, input: "db slow", expected: {severity: SEV1}, tags: [db]}
  - {id: crash, input: crash, expected: {severity: SEV1}}
  - {id: coin, input: coin, expected: {severity: SEV1}}
"""


def make_project(
    write: WriteFile, module_name: str, *, callable_name: str = "run", extra: str = ""
) -> Path:
    write(f"{module_name}.py", APP)
    write("cases.yaml", CASES)
    return write(
        "costless.yaml",
        f"""
        version: 1
        target: {{type: python, callable: "{module_name}:{callable_name}", timeout_s: 1}}
        datasets: [{{path: cases.yaml}}]
        run: {{repeats: 4, concurrency: 8}}
        scorers:
          - {{type: json_schema, schema: {{type: object, required: [severity]}}}}
          - {{type: exact_match, path: severity, weight: 3}}
        {extra}
        """,
    )


def run(path: Path, **kwargs: object) -> RunResult:
    return asyncio.run(run_suite(load_config(path), **kwargs))  # type: ignore[arg-type]


def test_end_to_end_aggregation(write: WriteFile, module_name: str) -> None:
    result = run(make_project(write, module_name))
    cases = {c.case_id: c for c in result.cases}

    assert result.metadata.repeats == 4
    assert result.metadata.datasets[0].name == "incidents"
    assert [c.case_id for c in result.cases] == [
        "incidents/down",
        "incidents/wrong",
        "incidents/crash",
        "incidents/coin",
    ]
    assert len(result.attempts) == 16

    assert cases["incidents/down"].pass_rate == 1.0
    assert cases["incidents/down"].quality_mean == 1.0

    # Schema passes (weight 1), severity wrong (weight 3): quality = 1/4.
    assert cases["incidents/wrong"].quality_mean == pytest.approx(0.25)
    assert cases["incidents/wrong"].pass_rate == 0.0
    assert cases["incidents/wrong"].quality_variance == 0.0

    assert cases["incidents/crash"].errors == 4
    assert cases["incidents/crash"].quality_mean == 0.0

    coin = cases["incidents/coin"]
    assert coin.flaky
    assert coin.pass_rate == 0.5
    assert coin.quality_mean == pytest.approx(0.625)  # mean of [1, .25, 1, .25]
    assert coin.quality_variance == pytest.approx(0.1875)  # sample variance of the same

    s = result.summary
    assert s.cases == 4
    assert s.attempts == 16
    assert s.quality_mean == pytest.approx((1 + 0.25 + 0 + 0.625) / 4)
    assert s.failure_rate == pytest.approx(10 / 16)
    assert s.error_rate == pytest.approx(4 / 16)
    assert s.flaky_cases == 1
    # Usage is reported by the app through the attempt context, even for crashes.
    assert s.input_tokens == 4 * (len("db down") + len("db slow") + len("crash") + len("coin"))
    assert s.output_tokens == 16 * 3


def test_error_attempts_record_the_exception(write: WriteFile, module_name: str) -> None:
    result = run(make_project(write, module_name))
    crash = next(a for a in result.attempts if a.case_id == "incidents/crash")
    assert crash.error == "RuntimeError: boom"
    assert crash.output is None
    assert crash.scores == ()


def test_timeout(write: WriteFile, module_name: str) -> None:
    config = make_project(write, module_name)
    write("cases.yaml", "- {id: slow, input: slow, expected: {severity: SEV1}}\n")
    result = run(config, repeats=1)
    assert result.attempts[0].error == "timed out after 1s"


def test_sync_callable_returning_json_value(write: WriteFile, module_name: str) -> None:
    result = run(make_project(write, module_name, callable_name="sync_run"), repeats=1)
    assert result.attempts[0].output == '{"severity": "SEV1"}'


def test_tag_filter_and_repeat_override(write: WriteFile, module_name: str) -> None:
    result = run(make_project(write, module_name), repeats=2, tags=["db"])
    assert [c.case_id for c in result.cases] == ["incidents/down", "incidents/wrong"]
    assert result.summary.attempts == 4


def test_ungradable_cases_fail_before_running(write: WriteFile, module_name: str) -> None:
    config = make_project(write, module_name)
    write("cases.yaml", "- {id: no-expected, input: x}\n")
    with pytest.raises(ConfigError, match=r"cases/no-expected: scorer .* needs an 'expected'"):
        run(config)


def test_case_without_any_scorer(write: WriteFile, module_name: str) -> None:
    write(f"{module_name}.py", APP)
    write("cases.yaml", "- {id: a, input: x}\n")
    config = write(
        "costless.yaml",
        f"""
        version: 1
        target: {{type: python, callable: "{module_name}:run"}}
        datasets: [{{path: cases.yaml}}]
        """,
    )
    with pytest.raises(ConfigError, match="no scorers configured"):
        run(config)


def test_unknown_target_module(write: WriteFile) -> None:
    write("cases.yaml", "- {id: a, input: x, scorers: [{type: regex, pattern: x}]}\n")
    config = write(
        "costless.yaml",
        """
        version: 1
        target: {type: python, callable: "does_not_exist_anywhere:run"}
        datasets: [{path: cases.yaml}]
        """,
    )
    with pytest.raises(ConfigError, match="cannot import target module"):
        run(config)


SUBPROCESS_SUT = """
import json, sys
req = json.load(sys.stdin)
if req["input"] == "fail":
    sys.stderr.write("something went wrong")
    sys.exit(3)
if req["input"] == "garbage":
    print("not json")
    sys.exit(0)
print(json.dumps({
    "output": {"severity": "SEV1", "case": req["case_id"]},
    "usage": [{"provider": "node", "model": "m", "input_tokens": 11, "output_tokens": 2}],
}))
"""


def test_subprocess_target(write: WriteFile) -> None:
    write("sut.py", SUBPROCESS_SUT)
    write(
        "cases.yaml",
        """
        - {id: ok, input: fine, expected: {severity: SEV1}}
        - {id: fail, input: fail, expected: {severity: SEV1}}
        - {id: garbage, input: garbage, expected: {severity: SEV1}}
        """,
    )
    config = write(
        "costless.yaml",
        f"""
        version: 1
        target: {{type: subprocess, command: [{json.dumps(sys.executable)}, sut.py]}}
        datasets: [{{path: cases.yaml}}]
        scorers: [{{type: exact_match, path: severity}}]
        """,
    )
    result = run(config, repeats=1)
    by_id = {a.case_id: a for a in result.attempts}

    assert by_id["cases/ok"].passed
    assert json.loads(by_id["cases/ok"].output or "")["case"] == "ok"
    assert by_id["cases/ok"].input_tokens == 11
    assert by_id["cases/fail"].error == "TargetError: exited with status 3: something went wrong"
    assert by_id["cases/garbage"].error is not None
    assert "stdout is not a JSON object" in by_id["cases/garbage"].error
    assert result.metadata.target.startswith("subprocess:")


def test_results_round_trip(write: WriteFile, module_name: str, tmp_path: Path) -> None:
    result = run(make_project(write, module_name), repeats=1)
    path = tmp_path / "out" / "run.json"
    write_run(result, path)

    assert read_run(path) == result
    assert json.loads(path.read_text())["schema_version"] == 1


def test_subprocess_timeout_kills_the_child(write: WriteFile, tmp_path: Path) -> None:
    marker = tmp_path / "survived"
    write(
        "sleepy.py",
        f"""
        import pathlib, time
        time.sleep(1.5)
        pathlib.Path({str(marker)!r}).write_text("still running")
        """,
    )
    write("cases.yaml", "- {id: a, input: x, scorers: [{type: regex, pattern: x}]}\n")
    config = write(
        "costless.yaml",
        f"""
        version: 1
        target:
          type: subprocess
          command: [{json.dumps(sys.executable)}, sleepy.py]
          timeout_s: 0.5
        datasets: [{{path: cases.yaml}}]
        """,
    )
    result = run(config, repeats=1)
    assert result.attempts[0].error == "timed out after 0.5s"

    time.sleep(2)  # longer than the child would have slept
    assert not marker.exists()
