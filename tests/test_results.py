import json
from pathlib import Path

import pytest

from costless.errors import ResultsError
from costless.results import read_run, run_json_schema


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{oops", "invalid JSON"),
        ('{"schema_version": 99}', "unsupported schema_version 99"),
        ('{"schema_version": 1}', "Field required"),
    ],
)
def test_read_run_rejects_bad_documents(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "run.json"
    path.write_text(content)
    with pytest.raises(ResultsError, match=message):
        read_run(path)


def test_read_run_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ResultsError, match="cannot read results"):
        read_run(tmp_path / "run.json")


def test_json_schema_describes_attempts() -> None:
    schema = run_json_schema()
    assert "attempts" in json.dumps(schema)
