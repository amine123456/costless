"""Read and write run.json documents."""

import json
from pathlib import Path

from pydantic import ValidationError

from costless.errors import ResultsError
from costless.models import RUN_SCHEMA_VERSION, RunResult


def write_run(result: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic: a crashed run never leaves a half-written baseline


def read_run(path: Path) -> RunResult:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"cannot read results {path}: {exc.strerror}"
        raise ResultsError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"{path}: invalid JSON: {exc.msg}"
        raise ResultsError(msg) from exc
    version = doc.get("schema_version") if isinstance(doc, dict) else None
    if version != RUN_SCHEMA_VERSION:
        msg = f"{path}: unsupported schema_version {version!r} (expected {RUN_SCHEMA_VERSION})"
        raise ResultsError(msg)
    try:
        return RunResult.model_validate(doc)
    except ValidationError as exc:
        msg = f"{path}: {exc}"
        raise ResultsError(msg) from exc


def run_json_schema() -> dict[str, object]:
    return RunResult.model_json_schema()
