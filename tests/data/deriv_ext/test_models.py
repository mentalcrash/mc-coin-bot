"""Tests for Coinalyze Extended Derivatives data models."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.data.deriv_ext.models import (
    AggFundingRecord,
    AggOIRecord,
    CVDRecord,
    DerivExtBatch,
    LiquidationRecord,
)


class TestAggOIRecord:
    """AggOIRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = AggOIRecord(
            date="2024-01-15",
            symbol="BTCUSDT.6",
            open=Decimal(50000),
            high=Decimal(52000),
            low=Decimal(49000),
            close=Decimal(51000),
        )
        assert record.symbol == "BTCUSDT.6"
        assert record.close == Decimal(51000)
        assert record.source == "coinalyze"
        assert record.date.tzinfo is not None

    def test_date_utc_enforce(self) -> None:
        """날짜 UTC 강제 적용."""
        record = AggOIRecord(
            date="2024-01-15",
            symbol="BTCUSDT.6",
            open=Decimal(1),
            high=Decimal(1),
            low=Decimal(1),
            close=Decimal(1),
        )
        assert record.date.tzinfo == UTC

    def test_date_from_unix_seconds(self) -> None:
        """Unix 초 timestamp → UTC datetime."""
        # 1705276800 = 2024-01-15 00:00:00 UTC
        record = AggOIRecord(
            date=1705276800,
            symbol="BTCUSDT.6",
            open=Decimal(1),
            high=Decimal(1),
            low=Decimal(1),
            close=Decimal(1),
        )
        assert record.date.year == 2024
        assert record.date.month == 1
        assert record.date.day == 15

    def test_date_from_unix_ms(self) -> None:
        """Unix 밀리초 timestamp도 자동 감지."""
        record = AggOIRecord(
            date=1705276800000,
            symbol="BTCUSDT.6",
            open=Decimal(1),
            high=Decimal(1),
            low=Decimal(1),
            close=Decimal(1),
        )
        assert record.date.year == 2024

    def test_frozen(self) -> None:
        """Frozen 모델은 수정 불가."""
        record = AggOIRecord(
            date="2024-01-15",
            symbol="BTCUSDT.6",
            open=Decimal(1),
            high=Decimal(1),
            low=Decimal(1),
            close=Decimal(1),
        )
        with pytest.raises(ValidationError):
            record.close = Decimal(99)  # type: ignore[misc]


class TestAggFundingRecord:
    """AggFundingRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = AggFundingRecord(
            date="2024-01-15",
            symbol="ETHUSDT.6",
            open=Decimal("0.0001"),
            high=Decimal("0.0003"),
            low=Decimal("-0.0001"),
            close=Decimal("0.0002"),
        )
        assert record.symbol == "ETHUSDT.6"
        assert record.close == Decimal("0.0002")
        assert record.source == "coinalyze"


class TestLiquidationRecord:
    """LiquidationRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = LiquidationRecord(
            date="2024-01-15",
            symbol="BTCUSDT.6",
            long_volume=Decimal(1500000),
            short_volume=Decimal(800000),
        )
        assert record.long_volume == Decimal(1500000)
        assert record.short_volume == Decimal(800000)
        assert record.source == "coinalyze"

    def test_frozen(self) -> None:
        """Frozen 제약 검증."""
        record = LiquidationRecord(
            date="2024-01-15",
            symbol="BTCUSDT.6",
            long_volume=Decimal(100),
            short_volume=Decimal(200),
        )
        with pytest.raises(ValidationError):
            record.long_volume = Decimal(999)  # type: ignore[misc]


class TestCVDRecord:
    """CVDRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = CVDRecord(
            date="2024-01-15",
            symbol="BTCUSDT.6",
            open=Decimal(100),
            high=Decimal(200),
            low=Decimal(50),
            close=Decimal(180),
            volume=Decimal(50000),
            buy_volume=Decimal(28000),
        )
        assert record.volume == Decimal(50000)
        assert record.buy_volume == Decimal(28000)
        assert record.source == "coinalyze"


class TestDerivExtBatch:
    """DerivExtBatch 모델 테스트."""

    def test_basic_batch(self) -> None:
        """기본 배치 생성."""
        records = (
            AggOIRecord(
                date="2024-01-15",
                symbol="BTCUSDT.6",
                open=Decimal(1),
                high=Decimal(1),
                low=Decimal(1),
                close=Decimal(1),
            ),
            AggOIRecord(
                date="2024-01-16",
                symbol="BTCUSDT.6",
                open=Decimal(2),
                high=Decimal(2),
                low=Decimal(2),
                close=Decimal(2),
            ),
        )
        batch = DerivExtBatch(
            source="coinalyze",
            data_type="btc_agg_oi",
            records=records,
        )
        assert batch.count == 2
        assert batch.is_empty is False

    def test_empty_batch(self) -> None:
        """빈 배치."""
        batch = DerivExtBatch(source="coinalyze", data_type="btc_agg_oi", records=())
        assert batch.count == 0
        assert batch.is_empty is True

    def test_fetched_at_utc(self) -> None:
        """fetched_at UTC 강제."""
        batch = DerivExtBatch(
            source="coinalyze",
            data_type="btc_agg_oi",
            records=(),
            fetched_at=datetime(2024, 1, 15),
        )
        assert batch.fetched_at.tzinfo == UTC

    def test_mixed_record_types(self) -> None:
        """여러 레코드 타입 혼합."""
        records = (
            AggOIRecord(
                date="2024-01-15",
                symbol="BTCUSDT.6",
                open=Decimal(1),
                high=Decimal(1),
                low=Decimal(1),
                close=Decimal(1),
            ),
            LiquidationRecord(
                date="2024-01-15",
                symbol="BTCUSDT.6",
                long_volume=Decimal(100),
                short_volume=Decimal(200),
            ),
        )
        batch = DerivExtBatch(
            source="coinalyze",
            data_type="mixed",
            records=records,
        )
        assert batch.count == 2
