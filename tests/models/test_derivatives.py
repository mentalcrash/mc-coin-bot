"""Tests for src/models/derivatives.py — Pydantic V2 derivatives models."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.models.derivatives import (
    DerivativesBatch,
    FundingRateRecord,
    LongShortRatioRecord,
    OpenInterestRecord,
    TakerRatioRecord,
)


class TestFundingRateRecord:
    def test_create_from_values(self) -> None:
        r = FundingRateRecord(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("42000.50"),
        )
        assert r.symbol == "BTC/USDT"
        assert r.funding_rate == Decimal("0.0001")

    def test_timestamp_from_unix_ms(self) -> None:
        ts_ms = int(datetime(2024, 6, 15, 8, 0, 0, tzinfo=UTC).timestamp() * 1000)
        r = FundingRateRecord(
            symbol="ETH/USDT",
            timestamp=ts_ms,
            funding_rate=Decimal("-0.0003"),
            mark_price=Decimal("3500.00"),
        )
        assert r.timestamp.year == 2024
        assert r.timestamp.month == 6
        assert r.timestamp.tzinfo is not None

    def test_frozen(self) -> None:
        r = FundingRateRecord(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal(42000),
        )
        with pytest.raises((TypeError, ValueError)):
            r.symbol = "ETH/USDT"  # type: ignore[misc]

    def test_negative_funding_rate_allowed(self) -> None:
        """Funding rate는 음수 가능."""
        r = FundingRateRecord(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            funding_rate=Decimal("-0.0005"),
            mark_price=Decimal(42000),
        )
        assert r.funding_rate < 0


class TestOpenInterestRecord:
    def test_create(self) -> None:
        r = OpenInterestRecord(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            sum_open_interest=Decimal(50000),
            sum_open_interest_value=Decimal(2100000000),
        )
        assert r.sum_open_interest == Decimal(50000)
        assert r.sum_open_interest_value == Decimal(2100000000)

    def test_ge_zero_constraint(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            OpenInterestRecord(
                symbol="BTC/USDT",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                sum_open_interest=Decimal(-1),
                sum_open_interest_value=Decimal(100),
            )


class TestLongShortRatioRecord:
    def test_create(self) -> None:
        r = LongShortRatioRecord(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            long_account=Decimal("0.55"),
            short_account=Decimal("0.45"),
            long_short_ratio=Decimal("1.22"),
        )
        assert r.long_account == Decimal("0.55")
        assert r.short_account == Decimal("0.45")

    def test_le_one_constraint(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            LongShortRatioRecord(
                symbol="BTC/USDT",
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                long_account=Decimal("1.5"),
                short_account=Decimal("0.45"),
                long_short_ratio=Decimal("3.33"),
            )


class TestTakerRatioRecord:
    def test_create(self) -> None:
        r = TakerRatioRecord(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            buy_vol=Decimal(1000),
            sell_vol=Decimal(800),
            buy_sell_ratio=Decimal("1.25"),
        )
        assert r.buy_sell_ratio == Decimal("1.25")

    def test_timestamp_unix_ms(self) -> None:
        ts = int(datetime(2024, 3, 1, tzinfo=UTC).timestamp() * 1000)
        r = TakerRatioRecord(
            symbol="BTC/USDT",
            timestamp=ts,
            buy_vol=Decimal(500),
            sell_vol=Decimal(500),
            buy_sell_ratio=Decimal("1.0"),
        )
        assert r.timestamp.year == 2024


class TestDerivativesBatch:
    def test_empty_batch(self) -> None:
        batch = DerivativesBatch(symbol="BTC/USDT")
        assert batch.is_empty
        assert batch.total_records == 0

    def test_batch_with_data(self) -> None:
        fr = FundingRateRecord(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal(42000),
        )
        oi = OpenInterestRecord(
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            sum_open_interest=Decimal(50000),
            sum_open_interest_value=Decimal(2000000000),
        )
        batch = DerivativesBatch(
            symbol="BTC/USDT",
            funding_rates=(fr,),
            open_interest=(oi,),
        )
        assert not batch.is_empty
        assert batch.total_records == 2

    def test_fetched_at_utc(self) -> None:
        batch = DerivativesBatch(symbol="BTC/USDT")
        assert batch.fetched_at.tzinfo is not None
