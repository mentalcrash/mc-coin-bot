"""Tests for Deribit Options data models."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.data.options.models import (
    DVolRecord,
    HistoricalVolRecord,
    MaxPainRecord,
    OptionsBatch,
    PutCallRatioRecord,
    TermStructureRecord,
)


class TestDVolRecord:
    """DVolRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = DVolRecord(
            date="2024-01-15",
            currency="BTC",
            close=Decimal("65.5"),
        )
        assert record.currency == "BTC"
        assert record.close == Decimal("65.5")
        assert record.source == "deribit"
        assert record.date.tzinfo is not None

    def test_full_ohlcv(self) -> None:
        """OHLCV 전체 필드."""
        record = DVolRecord(
            date="2024-01-15",
            currency="ETH",
            open=Decimal("60.0"),
            high=Decimal("70.0"),
            low=Decimal("58.0"),
            close=Decimal("65.0"),
            volume=Decimal("100.0"),
        )
        assert record.open == Decimal("60.0")
        assert record.high == Decimal("70.0")
        assert record.volume == Decimal("100.0")

    def test_date_utc_enforce(self) -> None:
        """날짜 UTC 강제 적용."""
        record = DVolRecord(date="2024-01-15", currency="BTC", close=Decimal("65.0"))
        assert record.date.tzinfo == UTC

    def test_date_from_unix_ms(self) -> None:
        """Unix 밀리초 timestamp → UTC datetime."""
        # 1705276800000 = 2024-01-15 00:00:00 UTC in ms
        record = DVolRecord(date=1705276800000, currency="BTC", close=Decimal("65.0"))
        assert record.date.year == 2024
        assert record.date.month == 1
        assert record.date.day == 15

    def test_date_from_unix_seconds(self) -> None:
        """Unix 초 timestamp → UTC datetime."""
        record = DVolRecord(date=1705276800, currency="BTC", close=Decimal("65.0"))
        assert record.date.year == 2024

    def test_frozen(self) -> None:
        """Frozen 모델은 수정 불가."""
        record = DVolRecord(date="2024-01-15", currency="BTC", close=Decimal("65.0"))
        with pytest.raises(ValidationError):
            record.close = Decimal("70.0")  # type: ignore[misc]

    def test_none_optional_fields(self) -> None:
        """Optional 필드 None."""
        record = DVolRecord(date="2024-01-15", currency="BTC", close=Decimal("65.0"))
        assert record.open is None
        assert record.high is None
        assert record.low is None
        assert record.volume is None


class TestPutCallRatioRecord:
    """PutCallRatioRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = PutCallRatioRecord(
            date="2024-01-15",
            currency="BTC",
            put_oi=Decimal(5000),
            call_oi=Decimal(8000),
            pc_ratio=Decimal("0.625"),
        )
        assert record.pc_ratio == Decimal("0.625")
        assert record.source == "deribit"

    def test_frozen(self) -> None:
        """Frozen 제약 검증."""
        record = PutCallRatioRecord(
            date="2024-01-15",
            currency="BTC",
            put_oi=Decimal(5000),
            call_oi=Decimal(8000),
            pc_ratio=Decimal("0.625"),
        )
        with pytest.raises(ValidationError):
            record.pc_ratio = Decimal("1.0")  # type: ignore[misc]


class TestHistoricalVolRecord:
    """HistoricalVolRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성 (일부 None)."""
        record = HistoricalVolRecord(
            date="2024-01-15",
            currency="BTC",
            vol_30d=Decimal("55.0"),
        )
        assert record.vol_30d == Decimal("55.0")
        assert record.vol_7d is None
        assert record.vol_365d is None

    def test_all_windows(self) -> None:
        """전체 윈도우 값."""
        record = HistoricalVolRecord(
            date="2024-01-15",
            currency="BTC",
            vol_7d=Decimal("40.0"),
            vol_30d=Decimal("50.0"),
            vol_60d=Decimal("55.0"),
            vol_90d=Decimal("58.0"),
            vol_120d=Decimal("60.0"),
            vol_180d=Decimal("62.0"),
            vol_365d=Decimal("65.0"),
        )
        assert record.vol_7d == Decimal("40.0")
        assert record.vol_365d == Decimal("65.0")


class TestTermStructureRecord:
    """TermStructureRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = TermStructureRecord(
            date="2024-01-15",
            currency="BTC",
            near_expiry="BTC-26JAN24",
            far_expiry="BTC-29MAR24",
            near_basis_pct=Decimal("1.2"),
            far_basis_pct=Decimal("3.5"),
            slope=Decimal("2.3"),
        )
        assert record.slope == Decimal("2.3")
        assert record.near_expiry == "BTC-26JAN24"

    def test_negative_slope_backwardation(self) -> None:
        """음수 slope (backwardation)."""
        record = TermStructureRecord(
            date="2024-01-15",
            currency="BTC",
            near_expiry="BTC-26JAN24",
            far_expiry="BTC-29MAR24",
            near_basis_pct=Decimal("3.5"),
            far_basis_pct=Decimal("1.2"),
            slope=Decimal("-2.3"),
        )
        assert record.slope < 0


class TestMaxPainRecord:
    """MaxPainRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = MaxPainRecord(
            date="2024-01-15",
            currency="BTC",
            expiry="26JAN24",
            max_pain_strike=Decimal(42000),
            total_oi=Decimal(15000),
        )
        assert record.max_pain_strike == Decimal(42000)
        assert record.expiry == "26JAN24"
        assert record.source == "deribit"


class TestOptionsBatch:
    """OptionsBatch 모델 테스트."""

    def test_basic_batch(self) -> None:
        """기본 배치 생성."""
        records = (
            DVolRecord(date="2024-01-15", currency="BTC", close=Decimal("65.0")),
            DVolRecord(date="2024-01-16", currency="BTC", close=Decimal("66.0")),
        )
        batch = OptionsBatch(
            source="deribit",
            data_type="btc_dvol",
            records=records,
        )
        assert batch.count == 2
        assert batch.is_empty is False

    def test_empty_batch(self) -> None:
        """빈 배치."""
        batch = OptionsBatch(source="deribit", data_type="btc_dvol", records=())
        assert batch.count == 0
        assert batch.is_empty is True

    def test_fetched_at_utc(self) -> None:
        """fetched_at UTC 강제."""
        batch = OptionsBatch(
            source="deribit",
            data_type="btc_dvol",
            records=(),
            fetched_at=datetime(2024, 1, 15),
        )
        assert batch.fetched_at.tzinfo == UTC

    def test_mixed_record_types(self) -> None:
        """여러 레코드 타입 혼합."""
        records = (
            DVolRecord(date="2024-01-15", currency="BTC", close=Decimal("65.0")),
            PutCallRatioRecord(
                date="2024-01-15",
                currency="BTC",
                put_oi=Decimal(5000),
                call_oi=Decimal(8000),
                pc_ratio=Decimal("0.625"),
            ),
        )
        batch = OptionsBatch(
            source="deribit",
            data_type="mixed",
            records=records,
        )
        assert batch.count == 2
