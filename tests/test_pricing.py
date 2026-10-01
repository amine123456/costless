from decimal import Decimal
from pathlib import Path

import pytest

from costless.errors import ConfigError
from costless.models import Usage
from costless.pricing import ModelPrice, PricingTable, to_report


def usage(model: str, i: int = 0, o: int = 0, cr: int = 0, cw: int = 0) -> Usage:
    return Usage(
        model=model, input_tokens=i, output_tokens=o, cache_read_tokens=cr, cache_write_tokens=cw
    )


def test_default_table_matches_published_anthropic_prices() -> None:
    # Values from https://platform.claude.com/docs/en/about-claude/pricing (2026-10-01).
    price = PricingTable.default().price_for("claude-opus-5-5")
    assert price == ModelPrice(
        input=Decimal(4), output=Decimal(20), cache_read=Decimal("0.20"), cache_write=Decimal(5)
    )
    haiku = PricingTable.default().price_for("claude-haiku-4-5")
    assert haiku is not None
    assert (haiku.input, haiku.output) == (Decimal(1), Decimal(5))


def test_default_table_matches_published_xai_prices() -> None:
    # Values from https://docs.x.ai/developers/pricing (2026-10-01), < 200k prompt tier.
    price = PricingTable.default().price_for("grok-4.7")
    assert price == ModelPrice(input=Decimal(2), output=Decimal(6), cache_read=Decimal("0.50"))
    # 1M uncached input + 1M cached input + 1M output
    cost = PricingTable.default().cost([usage("grok-4.7", i=10**6, cr=10**6, o=10**6)])
    assert cost.usd == Decimal("8.50")


def test_cost_combines_all_token_kinds() -> None:
    table = PricingTable.default()
    # 1M input @ $4 + 0.5M output @ $20 + 2M cache reads @ $0.20 + 0.1M cache writes @ $5
    breakdown = table.cost(
        [usage("claude-opus-5-5", i=1_000_000, o=500_000, cr=2_000_000, cw=100_000)]
    )
    assert breakdown.usd == Decimal("4") + Decimal("10") + Decimal("0.4") + Decimal("0.5")
    assert breakdown.complete


def test_small_costs_do_not_drift() -> None:
    table = PricingTable.default()
    one_call = usage("claude-haiku-4-5", i=137, o=59)  # $0.000137 + $0.000295
    breakdown = table.cost([one_call] * 10_000)
    assert breakdown.usd == Decimal("4.32")
    assert to_report(table.cost([one_call]).usd) == 0.000432


def test_cache_prices_default_to_input_price() -> None:
    table = PricingTable({"local": ModelPrice(input=Decimal(1), output=Decimal(2))})
    assert table.cost([usage("local", cr=1_000_000, cw=1_000_000)]).usd == Decimal(2)


def test_unknown_models_are_reported_not_guessed() -> None:
    breakdown = PricingTable.default().cost(
        [usage("claude-haiku-4-5", i=1_000_000), usage("grok-0", i=10), usage("grok-0", o=1)]
    )
    assert breakdown.usd == Decimal(1)
    assert breakdown.unpriced_models == frozenset({"grok-0"})
    assert not breakdown.complete


def test_globs_match_most_specific_first_and_exact_wins() -> None:
    table = PricingTable(
        {
            "grok-*": ModelPrice(input=Decimal(1), output=Decimal(1)),
            "grok-4-fast*": ModelPrice(input=Decimal(2), output=Decimal(2)),
            "grok-4-fast-x": ModelPrice(input=Decimal(3), output=Decimal(3)),
        }
    )
    assert table.price_for("grok-3") == ModelPrice(input=Decimal(1), output=Decimal(1))
    assert table.price_for("grok-4-fast-reasoning") == ModelPrice(
        input=Decimal(2), output=Decimal(2)
    )
    assert table.price_for("grok-4-fast-x") == ModelPrice(input=Decimal(3), output=Decimal(3))
    assert table.price_for("gpt-x") is None


def test_merged_overrides_take_precedence() -> None:
    cheap = ModelPrice(input=Decimal(0), output=Decimal(0))
    table = PricingTable.default().merged({"claude-opus-5-5": cheap, "grok-4": cheap})
    assert table.price_for("claude-opus-5-5") == cheap
    assert table.price_for("grok-4") == cheap
    assert table.price_for("claude-haiku-4-5") is not None


def test_from_file(tmp_path: Path) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text("version: 1\nmodels:\n  my-model: {input: 0.5, output: 1.5}\n")
    price = PricingTable.from_file(path).price_for("my-model")
    assert price is not None
    assert price.output == Decimal("1.5")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("models: {m: {input: -1, output: 1}}\n", "invalid pricing table"),
        ("models: {m: {input: 1}}\n", "invalid pricing table"),
        ("currency: EUR\nmodels: {}\n", "only USD"),
        ("models: [\n", "invalid YAML"),
    ],
)
def test_invalid_pricing_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text(content)
    with pytest.raises(ConfigError, match=message):
        PricingTable.from_file(path)


def test_missing_pricing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read pricing file"):
        PricingTable.from_file(tmp_path / "nope.yaml")
