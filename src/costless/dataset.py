"""Load versioned eval datasets from YAML or JSONL files.

YAML layout::

    dataset: incident-summaries   # optional, defaults to the file stem
    version: "3"                  # optional human-readable version
    cases:
      - id: inc-001
        input: {...}
        expected: {...}
        tags: [database]

JSONL layout: one case object per line; the dataset name is the file stem.

Every dataset also gets a content hash, so a run records exactly which cases it used.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from costless.errors import DatasetError
from costless.models import Case, DatasetInfo

_YAML_SUFFIXES = {".yaml", ".yml"}
_JSONL_SUFFIXES = {".jsonl"}


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str | None
    path: Path
    sha256: str
    cases: tuple[Case, ...]

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name, version=self.version, sha256=self.sha256, cases=len(self.cases)
        )

    def filter_tags(self, tags: Iterable[str]) -> "Dataset":
        """Keep only cases carrying at least one of ``tags``. An empty filter keeps everything."""
        wanted = set(tags)
        if not wanted:
            return self
        kept = tuple(c for c in self.cases if wanted.intersection(c.tags))
        return Dataset(self.name, self.version, self.path, self.sha256, kept)


def load_dataset(path: Path) -> Dataset:
    if not path.is_file():
        msg = f"dataset not found: {path}"
        raise DatasetError(msg)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")

    suffix = path.suffix.lower()
    if suffix in _YAML_SUFFIXES:
        name, version, records = _parse_yaml(text, path)
    elif suffix in _JSONL_SUFFIXES:
        name, version, records = path.stem, None, _parse_jsonl(text, path)
    else:
        msg = f"{path}: unsupported dataset format {suffix!r} (use .yaml, .yml or .jsonl)"
        raise DatasetError(msg)

    cases = tuple(_to_case(record, path, index) for index, record in enumerate(records))
    if not cases:
        msg = f"{path}: dataset has no cases"
        raise DatasetError(msg)
    _check_unique_ids(cases, path)
    return Dataset(name=name, version=version, path=path, sha256=digest, cases=cases)


def _parse_yaml(text: str, path: Path) -> tuple[str, str | None, list[Any]]:
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"{path}: invalid YAML: {exc}"
        raise DatasetError(msg) from exc

    if isinstance(doc, list):
        return path.stem, None, doc
    if not isinstance(doc, dict) or not isinstance(doc.get("cases"), list):
        msg = f"{path}: expected a list of cases or a mapping with a 'cases' list"
        raise DatasetError(msg)
    unknown = set(doc) - {"dataset", "version", "cases"}
    if unknown:
        msg = f"{path}: unknown top-level keys: {', '.join(sorted(unknown))}"
        raise DatasetError(msg)
    version = doc.get("version")
    return (
        str(doc.get("dataset") or path.stem),
        None if version is None else str(version),
        doc["cases"],
    )


def _parse_jsonl(text: str, path: Path) -> list[Any]:
    records: list[Any] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            msg = f"{path}:{lineno}: invalid JSON: {exc.msg}"
            raise DatasetError(msg) from exc
    return records


def _to_case(record: Any, path: Path, index: int) -> Case:  # noqa: ANN401
    try:
        return Case.model_validate(record)
    except ValidationError as exc:
        where = record.get("id", f"#{index}") if isinstance(record, dict) else f"#{index}"
        msg = f"{path}: case {where}: {exc}"
        raise DatasetError(msg) from exc


def _check_unique_ids(cases: tuple[Case, ...], path: Path) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            msg = f"{path}: duplicate case id {case.id!r}"
            raise DatasetError(msg)
        seen.add(case.id)
