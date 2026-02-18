"""Tests for CoinGecko data models."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.data.macro.models import CoinGeckoDefiRecord, CoinGeckoGlobalRecord


class TestCoinGeckoGlobalRecord:
    """CoinGeckoGlobalRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = CoinGeckoGlobalRecord(
            date="2026-02-18T12:00:00",
            btc_dominance=Decimal("58.5"),
            eth_dominance=Decimal("12.3"),
            total_market_cap_usd=Decimal(2500000000000),
            total_volume_usd=Decimal(150000000000),
            active_cryptocurrencies=15000,
        )
        assert record.btc_dominance == Decimal("58.5")
        assert record.source == "coingecko"
        assert record.active_cryptocurrencies == 15000

    def test_date_utc_enforce(self) -> None:
        """날짜 UTC 강제 적용."""
        record = CoinGeckoGlobalRecord(
            date="2026-02-18",
            btc_dominance="58.5",
            eth_dominance="12.3",
            total_market_cap_usd="2500000000000",
            total_volume_usd="150000000000",
            active_cryptocurrencies=15000,
        )
        assert record.date.tzinfo == UTC

    def test_date_from_unix(self) -> None:
        """Unix timestamp → UTC datetime."""
        record = CoinGeckoGlobalRecord(
            date=1705276800,
            btc_dominance="58.5",
            eth_dominance="12.3",
            total_market_cap_usd="2500000000000",
            total_volume_usd="150000000000",
            active_cryptocurrencies=15000,
        )
        assert record.date.year == 2024

    def test_frozen(self) -> None:
        """Frozen 모델은 수정 불가."""
        record = CoinGeckoGlobalRecord(
            date="2026-02-18",
            btc_dominance="58.5",
            eth_dominance="12.3",
            total_market_cap_usd="2500000000000",
            total_volume_usd="150000000000",
            active_cryptocurrencies=15000,
        )
        with pytest.raises(ValidationError):
            record.btc_dominance = Decimal("60.0")  # type: ignore[misc]


class TestCoinGeckoDefiRecord:
    """CoinGeckoDefiRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = CoinGeckoDefiRecord(
            date="2026-02-18T12:00:00",
            defi_market_cap=Decimal(150000000000),
            defi_to_eth_ratio=Decimal("2.5"),
            defi_dominance=Decimal("4.2"),
        )
        assert record.defi_dominance == Decimal("4.2")
        assert record.source == "coingecko"

    def test_date_utc(self) -> None:
        """날짜 UTC 강제."""
        record = CoinGeckoDefiRecord(
            date=datetime(2026, 2, 18),
            defi_market_cap="150000000000",
            defi_to_eth_ratio="2.5",
            defi_dominance="4.2",
        )
        assert record.date.tzinfo == UTC
