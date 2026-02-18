"""Tests for Hyperliquid data models."""

from datetime import UTC
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.data.deriv_ext.models import HLAssetContextRecord, HLPredictedFundingRecord


class TestHLAssetContextRecord:
    """HLAssetContextRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = HLAssetContextRecord(
            date="2026-02-18T12:00:00",
            coin="BTC",
            mark_price=Decimal("95000.5"),
            open_interest=Decimal(500000000),
            funding=Decimal("0.0001"),
            premium=Decimal("0.002"),
            day_ntl_vlm=Decimal(5000000000),
        )
        assert record.coin == "BTC"
        assert record.mark_price == Decimal("95000.5")
        assert record.source == "hyperliquid"

    def test_premium_none(self) -> None:
        """premium은 선택 필드."""
        record = HLAssetContextRecord(
            date="2026-02-18T12:00:00",
            coin="ETH",
            mark_price=Decimal(3200),
            open_interest=Decimal(100000000),
            funding=Decimal("-0.0005"),
            day_ntl_vlm=Decimal(2000000000),
        )
        assert record.premium is None

    def test_date_utc(self) -> None:
        """날짜 UTC 강제."""
        record = HLAssetContextRecord(
            date="2026-02-18",
            coin="BTC",
            mark_price="95000",
            open_interest="500000000",
            funding="0.0001",
            day_ntl_vlm="5000000000",
        )
        assert record.date.tzinfo == UTC

    def test_frozen(self) -> None:
        """Frozen 모델은 수정 불가."""
        record = HLAssetContextRecord(
            date="2026-02-18",
            coin="BTC",
            mark_price="95000",
            open_interest="500000000",
            funding="0.0001",
            day_ntl_vlm="5000000000",
        )
        with pytest.raises(ValidationError):
            record.coin = "ETH"  # type: ignore[misc]


class TestHLPredictedFundingRecord:
    """HLPredictedFundingRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = HLPredictedFundingRecord(
            date="2026-02-18T12:00:00",
            coin="BTC",
            venue="Binance",
            predicted_funding=Decimal("0.0002"),
        )
        assert record.venue == "Binance"
        assert record.predicted_funding == Decimal("0.0002")
        assert record.source == "hyperliquid"

    def test_date_utc(self) -> None:
        """날짜 UTC 강제."""
        record = HLPredictedFundingRecord(
            date=1705276800,
            coin="ETH",
            venue="Bybit",
            predicted_funding="0.0001",
        )
        assert record.date.tzinfo == UTC
        assert record.date.year == 2024
