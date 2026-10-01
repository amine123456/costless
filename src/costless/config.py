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
from costless.pricing import ModelPrice, PricingTable
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


class PricingSettings(_Strict):
    file: str | None = Field(
        default=None, description="Extra pricing table, merged over the packaged defaults."
    )
    models: dict[str, ModelPrice] = Field(
        default_factory=dict, description="Inline prices; highest precedence."
    )
    strict: bool = Field(
        default=True, description="Fail the run when a model call cannot be priced."
    )


class BudgetSettings(_Strict):
    max_run_usd: float | None = Field(
        default=None,
        gt=0,
        description="Hard cap for the whole run. Once spent, remaining attempts are skipped.",
    )
    max_case_usd: float | None = Field(
        default=None, gt=0, description="Upper bound on the mean cost of one attempt."
    )


class Config(_Strict):
    version: Literal[1]
    target: TargetSpec
    datasets: tuple[DatasetRef, ...] = Field(min_length=1)
    run: RunSettings = RunSettings()
    scorers: tuple[ScorerSpec, ...] = Field(
        default=(), description="Scorers applied to every case, in addition to per-case scorers."
    )
    pricing: PricingSettings = PricingSettings()
    budget: BudgetSettings = BudgetSettings()


@dataclass(frozen=True)
class LoadedConfig:
    config: Config
    path: Path
    sha256: str

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def pricing_table(self) -> PricingTable:
        settings = self.config.pricing
        table = PricingTable.default()
        if settings.file is not None:
            table = table.merged(PricingTable.from_file(self.resolve(settings.file)).prices)
        return table.merged(settings.models)

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
