"""costless.yaml: what to run, against which datasets, and how to score it.

Relative paths in the config are resolved against the directory that contains it.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from costless.errors import ConfigError
from costless.specs import ScorerSpec

DEFAULT_CONFIG_PATH = Path("costless.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PythonTargetSpec(_Strict):
    """An in-process Python callable, ``module.path:function``. Sync or async."""

    type: Literal["python"]
    callable: str = Field(pattern=r"^[\w.]+:[\w.]+$")
    timeout_s: float = Field(default=60.0, gt=0)


class SubprocessTargetSpec(_Strict):
    """Any executable speaking JSON over stdin/stdout, one process per attempt."""

    type: Literal["subprocess"]
    command: tuple[str, ...] = Field(min_length=1)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = Field(default=60.0, gt=0)


TargetSpec = Annotated[PythonTargetSpec | SubprocessTargetSpec, Field(discriminator="type")]


class DatasetRef(_Strict):
    path: str
    tags: tuple[str, ...] = Field(default=(), description="Only run cases with one of these tags.")


class RunSettings(_Strict):
    repeats: int = Field(default=5, ge=1, le=100)
    concurrency: int = Field(default=4, ge=1, le=64)


class Config(_Strict):
    version: Literal[1]
    target: TargetSpec
    datasets: tuple[DatasetRef, ...] = Field(min_length=1)
    run: RunSettings = RunSettings()
    scorers: tuple[ScorerSpec, ...] = Field(
        default=(), description="Scorers applied to every case, in addition to per-case scorers."
    )


@dataclass(frozen=True)
class LoadedConfig:
    config: Config
    path: Path
    sha256: str

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.base_dir / candidate


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> LoadedConfig:
    if not path.is_file():
        msg = f"config not found: {path}"
        raise ConfigError(msg)
    raw = path.read_bytes()
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        msg = f"{path}: invalid YAML: {exc}"
        raise ConfigError(msg) from exc
    try:
        config = Config.model_validate(doc)
    except ValidationError as exc:
        msg = f"{path}: {exc}"
        raise ConfigError(msg) from exc
    return LoadedConfig(config=config, path=path.resolve(), sha256=hashlib.sha256(raw).hexdigest())
