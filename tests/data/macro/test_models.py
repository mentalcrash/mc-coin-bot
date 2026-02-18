"""Tests for macro data models (FREDObservationRecord, YFinanceRecord, MacroBatch)."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.data.macro.models import FREDObservationRecord, MacroBatch, YFinanceRecord


class TestFREDObservationRecord:
    """FREDObservationRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = FREDObservationRecord(
            date="2024-01-15",
            series_id="DTWEXBGS",
            value="123.45",
        )
        assert record.series_id == "DTWEXBGS"
        assert record.value == Decimal("123.45")
        assert record.source == "fred"
        assert record.date.tzinfo is not None

    def test_fred_dot_value_becomes_none(self) -> None:
        """FRED "." 값은 None으로 변환."""
        record = FREDObservationRecord(
            date="2024-01-15",
            series_id="DGS10",
            value=".",
        )
        assert record.value is None

    def test_none_value(self) -> None:
        """None 값 그대로 유지."""
        record = FREDObservationRecord(
            date="2024-01-15",
            series_id="DGS10",
            value=None,
        )
        assert record.value is None

    def test_decimal_value(self) -> None:
        """Decimal 값 직접 전달."""
        record = FREDObservationRecord(
            date="2024-01-15",
            series_id="VIXCLS",
            value=Decimal("18.5"),
        )
        assert record.value == Decimal("18.5")

    def test_date_utc_enforce(self) -> None:
        """날짜 UTC 강제 적용."""
        record = FREDObservationRecord(
            date="2024-01-15",
            series_id="DGS10",
            value="4.2",
        )
        assert record.date.tzinfo == UTC

    def test_date_from_unix(self) -> None:
        """Unix timestamp → UTC datetime."""
        record = FREDObservationRecord(
            date=1705276800,
            series_id="DGS10",
            value="4.2",
        )
        assert record.date.year == 2024

    def test_frozen(self) -> None:
        """Frozen 모델은 수정 불가."""
        record = FREDObservationRecord(
            date="2024-01-15",
            series_id="DGS10",
            value="4.2",
        )
        with pytest.raises(ValidationError):
            record.value = Decimal("5.0")  # type: ignore[misc]


class TestYFinanceRecord:
    """YFinanceRecord 모델 테스트."""

    def test_basic_creation(self) -> None:
        """기본 레코드 생성."""
        record = YFinanceRecord(
            date="2024-01-15",
            ticker="SPY",
            open=Decimal("470.0"),
            high=Decimal("475.0"),
            low=Decimal("468.0"),
            close=Decimal("473.5"),
            volume=Decimal(50000000),
        )
        assert record.ticker == "SPY"
        assert record.close == Decimal("473.5")
        assert record.source == "yfinance"

    def test_date_utc(self) -> None:
        """날짜 UTC 강제 적용."""
        record = YFinanceRecord(
            date="2024-01-15",
            ticker="QQQ",
            open="400",
            high="405",
            low="398",
            close="403",
            volume="30000000",
        )
        assert record.date.tzinfo == UTC


class TestMacroBatch:
    """MacroBatch 모델 테스트."""

    def test_basic_batch(self) -> None:
        """기본 배치 생성."""
        records = (
            FREDObservationRecord(date="2024-01-15", series_id="DGS10", value="4.2"),
            FREDObservationRecord(date="2024-01-16", series_id="DGS10", value="4.1"),
        )
        batch = MacroBatch(
            source="fred",
            data_type="dgs10",
            records=records,
        )
        assert batch.count == 2
        assert batch.is_empty is False

    def test_empty_batch(self) -> None:
        """빈 배치."""
        batch = MacroBatch(source="fred", data_type="dxy", records=())
        assert batch.count == 0
        assert batch.is_empty is True

    def test_fetched_at_utc(self) -> None:
        """fetched_at UTC 강제."""
        batch = MacroBatch(
            source="fred",
            data_type="dxy",
            records=(),
            fetched_at=datetime(2024, 1, 15),
        )
        assert batch.fetched_at.tzinfo == UTC
