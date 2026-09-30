import pytest

from costless.config import PythonTargetSpec, SubprocessTargetSpec, load_config
from costless.errors import ConfigError
from tests.conftest import WriteFile


def test_minimal_config_gets_defaults(write: WriteFile) -> None:
    path = write(
        "costless.yaml",
        """
        version: 1
        target: {type: python, callable: "app.main:run"}
        datasets: [{path: data/cases.yaml}]
        """,
    )
    loaded = load_config(path)

    assert isinstance(loaded.config.target, PythonTargetSpec)
    assert loaded.config.run.repeats == 5
    assert loaded.config.run.concurrency == 4
    assert loaded.resolve("data/cases.yaml") == path.parent / "data/cases.yaml"
    assert len(loaded.sha256) == 64


def test_subprocess_target(write: WriteFile) -> None:
    loaded = load_config(
        write(
            "costless.yaml",
            """
            version: 1
            target:
              type: subprocess
              command: [node, dist/cli.js, eval-target]
              env: {LOG_LEVEL: error}
              timeout_s: 120
            datasets: [{path: cases.yaml, tags: [smoke]}]
            run: {repeats: 3, concurrency: 2}
            scorers:
              - {type: json_schema, schema: schema.json}
            """,
        )
    )
    target = loaded.config.target
    assert isinstance(target, SubprocessTargetSpec)
    assert target.command == ("node", "dist/cli.js", "eval-target")
    assert loaded.config.datasets[0].tags == ("smoke",)
    assert loaded.config.scorers[0].type == "json_schema"


BASE = "version: 1\ntarget: {type: python, callable: 'a:b'}\ndatasets: [{path: x}]\n"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("version: 2\ntarget: {type: python, callable: 'a:b'}\ndatasets: [{path: x}]\n", "version"),
        ("version: 1\ntarget: {type: rpc}\ndatasets: [{path: x}]\n", "target"),
        (
            "version: 1\ntarget: {type: python, callable: 'nocolon'}\ndatasets: [{path: x}]\n",
            "callable",
        ),
        ("version: 1\ntarget: {type: python, callable: 'a:b'}\ndatasets: []\n", "datasets"),
        (BASE + "run: {repeats: 0}\n", "repeats"),
        (BASE + "retries: 3\n", "retries"),
        ("version: [\n", "invalid YAML"),
    ],
)
def test_invalid_config(write: WriteFile, content: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(write("costless.yaml", content))


def test_missing_config(write: WriteFile) -> None:
    path = write("other.yaml", "x: 1\n")
    with pytest.raises(ConfigError, match="config not found"):
        load_config(path.with_name("costless.yaml"))
