import textwrap
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from costless.models import Case

WriteFile = Callable[[str, str], Path]


@pytest.fixture
def write(tmp_path: Path) -> WriteFile:
    """Write a dedented file under tmp_path and return its path."""

    def _write(name: str, content: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def module_name() -> str:
    """A unique importable module name, so tests never share sys.modules entries."""
    return f"sut_{uuid.uuid4().hex}"


def make_case(**overrides: object) -> Case:
    fields: dict[str, object] = {"id": "c1", "input": "in"}
    fields.update(overrides)
    return Case.model_validate(fields)
