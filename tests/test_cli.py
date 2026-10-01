import json
import re

from typer.testing import CliRunner

from costless import __version__
from costless.cli import app
from tests.conftest import WriteFile

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "Usage" in result.output


def _project(write: WriteFile, module_name: str) -> str:
    write(f"{module_name}.py", "def run(text):\n    return text.upper()\n")
    write(
        "cases.yaml",
        """
        dataset: shout
        version: "2"
        cases:
          - {id: a, input: hi, expected: HI}
          - {id: b, input: yo, expected: nope}
        """,
    )
    return str(
        write(
            "costless.yaml",
            f"""
            version: 1
            target: {{type: python, callable: "{module_name}:run"}}
            datasets: [{{path: cases.yaml}}]
            run: {{repeats: 2}}
            scorers: [{{type: exact_match}}]
            """,
        )
    )


def test_validate(write: WriteFile, module_name: str) -> None:
    result = runner.invoke(app, ["validate", "-c", _project(write, module_name)])

    assert result.exit_code == 0, result.output
    assert "dataset shout v2: 2 cases" in result.stdout
    assert "ok: 2 cases ready to run" in result.stdout


def test_run_writes_results(write: WriteFile, module_name: str) -> None:
    config = _project(write, module_name)
    out = write("placeholder", "").with_name("run.json")

    result = runner.invoke(app, ["run", "-c", config, "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert re.search(r"quality \(mean\)\s+0\.500", result.stdout)
    assert "shout/b: pass 0%" in result.stdout
    assert json.loads(out.read_text())["summary"]["attempts"] == 4


def test_config_errors_exit_with_code_2(write: WriteFile) -> None:
    result = runner.invoke(app, ["validate", "-c", str(write("costless.yaml", "version: 9\n"))])

    assert result.exit_code == 2
    assert "error:" in result.stderr


def test_schema_command() -> None:
    result = runner.invoke(app, ["schema"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["title"] == "RunResult"
