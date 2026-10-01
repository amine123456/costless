"""Token pricing: turn recorded usage into money.

Prices are USD per million tokens. The packaged table (``data/pricing.yaml``)
covers first-party Anthropic models; projects extend or override it from
costless.yaml. Model names match exactly first, then against glob keys such as
``my-fine-tune-*``; the most specific glob (longest pattern) wins.

Money is computed with ``Decimal`` so that summing thousands of tiny per-call
costs does not drift, and rounded to 1e-9 USD only when reported.
"""

import fnmatch
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from costless.errors import ConfigError
from costless.models import Usage

_PER_TOKEN = Decimal(1_000_000)
_REPORT_QUANTUM = Decimal("0.000000001")


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)
    cache_read: Decimal | None = Field(default=None, ge=0, description="Defaults to input.")
    cache_write: Decimal | None = Field(default=None, ge=0, description="Defaults to input.")

    def cost(self, usage: Usage) -> Decimal:
        cache_read = self.input if self.cache_read is None else self.cache_read
        cache_write = self.input if self.cache_write is None else self.cache_write
        total = (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * cache_read
            + usage.cache_write_tokens * cache_write
        )
        return total / _PER_TOKEN


class _PricingFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = 1
    currency: str = "USD"
    models: dict[str, ModelPrice]


@dataclass(frozen=True)
class CostBreakdown:
    usd: Decimal
    unpriced_models: frozenset[str]

    @property
    def complete(self) -> bool:
        return not self.unpriced_models


class PricingTable:
    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._exact = {k: v for k, v in prices.items() if not _is_glob(k)}
        self._globs = sorted(
            ((k, v) for k, v in prices.items() if _is_glob(k)),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )

    @classmethod
    def default(cls) -> "PricingTable":
        text = resources.files("costless.data").joinpath("pricing.yaml").read_text("utf-8")
        return cls(_parse(yaml.safe_load(text), "packaged pricing.yaml").models)

    @classmethod
    def from_file(cls, path: Path) -> "PricingTable":
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            msg = f"cannot read pricing file {path}: {exc.strerror}"
            raise ConfigError(msg) from exc
        except yaml.YAMLError as exc:
            msg = f"{path}: invalid YAML: {exc}"
            raise ConfigError(msg) from exc
        return cls(_parse(doc, str(path)).models)

    @property
    def prices(self) -> dict[str, ModelPrice]:
        return {**self._exact, **dict(self._globs)}

    def merged(self, overrides: Mapping[str, ModelPrice]) -> "PricingTable":
        return PricingTable({**self.prices, **overrides})

    def price_for(self, model: str) -> ModelPrice | None:
        if model in self._exact:
            return self._exact[model]
        for pattern, price in self._globs:
            if fnmatch.fnmatchcase(model, pattern):
                return price
        return None

    def cost(self, usages: Iterable[Usage]) -> CostBreakdown:
        total = Decimal(0)
        unpriced: set[str] = set()
        for usage in usages:
            price = self.price_for(usage.model)
            if price is None:
                unpriced.add(usage.model)
            else:
                total += price.cost(usage)
        return CostBreakdown(usd=total, unpriced_models=frozenset(unpriced))


def to_report(usd: Decimal) -> float:
    """Round a Decimal amount for run.json."""
    return float(usd.quantize(_REPORT_QUANTUM))


def _is_glob(key: str) -> bool:
    return any(ch in key for ch in "*?[")


def _parse(doc: Any, where: str) -> _PricingFile:  # noqa: ANN401
    try:
        parsed = _PricingFile.model_validate(doc)
    except ValidationError as exc:
        msg = f"{where}: invalid pricing table: {exc}"
        raise ConfigError(msg) from exc
    if parsed.currency != "USD":
        msg = f"{where}: only USD pricing is supported, got {parsed.currency!r}"
        raise ConfigError(msg)
    return parsed
