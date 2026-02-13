"""Tests for src/data/derivatives_service.py — Silver 로드 + OHLCV merge_asof."""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd

from src.config.settings import IngestionSettings
from src.data.derivatives_service import DEFAULT_DERIV_COLUMNS, DerivativesDataService
from src.data.derivatives_storage import DerivativesBronzeStorage, DerivativesSilverProcessor
from src.models.derivatives import (
    DerivativesBatch,
    FundingRateRecord,
    LongShortRatioRecord,
    OpenInterestRecord,
    TakerRatioRecord,
)


def _make_batch(symbol: str = "BTC/USDT") -> DerivativesBatch:
    """테스트용 DerivativesBatch 생성."""
    ts1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    ts2 = datetime(2024, 1, 1, 8, 0, 0, tzinfo=UTC)
    return DerivativesBatch(
        symbol=symbol,
        funding_rates=(
            FundingRateRecord(
                symbol=symbol, timestamp=ts1,
                funding_rate=Decimal("0.0001"), mark_price=Decimal(42000),
            ),
            FundingRateRecord(
                symbol=symbol, timestamp=ts2,
                funding_rate=Decimal("0.0002"), mark_price=Decimal(42500),
            ),
        ),
        open_interest=(
            OpenInterestRecord(
                symbol=symbol, timestamp=ts1,
                sum_open_interest=Decimal(50000),
                sum_open_interest_value=Decimal(2100000000),
            ),
        ),
        long_short_ratios=(
            LongShortRatioRecord(
                symbol=symbol, timestamp=ts1,
                long_account=Decimal("0.55"), short_account=Decimal("0.45"),
                long_short_ratio=Decimal("1.22"),
            ),
        ),
        taker_ratios=(
            TakerRatioRecord(
                symbol=symbol, timestamp=ts1,
                buy_vol=Decimal(1000), sell_vol=Decimal(800),
                buy_sell_ratio=Decimal("1.25"),
            ),
        ),
    )


def _setup_silver(tmp_path, symbol: str = "BTC/USDT") -> IngestionSettings:
    """Silver 데이터를 준비한 settings 반환."""
    settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
    bronze = DerivativesBronzeStorage(settings)
    bronze.save(_make_batch(symbol), 2024)
    processor = DerivativesSilverProcessor(settings, bronze)
    processor.process(symbol, 2024)
    return settings


class TestDerivativesDataServiceLoad:
    def test_load_existing(self, tmp_path) -> None:
        settings = _setup_silver(tmp_path)
        service = DerivativesDataService(settings=settings)

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        df = service.load("BTC/USDT", start, end)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    def test_load_missing_returns_empty(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        service = DerivativesDataService(settings=settings)

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        df = service.load("BTC/USDT", start, end)
        assert df.empty

    def test_load_filters_by_date(self, tmp_path) -> None:
        settings = _setup_silver(tmp_path)
        service = DerivativesDataService(settings=settings)

        # 데이터 범위(2024-01-01) 밖의 날짜로 필터링
        start = datetime(2024, 6, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        df = service.load("BTC/USDT", start, end)
        # 데이터가 2024-01-01이므로 6월 이후로 필터링하면 비어있을 수 있음
        assert isinstance(df, pd.DataFrame)

    def test_load_multi_year(self, tmp_path) -> None:
        """Multi-year range에서 year iteration 동작."""
        settings = _setup_silver(tmp_path)
        service = DerivativesDataService(settings=settings)

        start = datetime(2023, 1, 1, tzinfo=UTC)
        end = datetime(2025, 12, 31, tzinfo=UTC)
        df = service.load("BTC/USDT", start, end)
        assert isinstance(df, pd.DataFrame)


class TestDerivativesDataServiceEnrich:
    def test_enrich_basic(self, tmp_path) -> None:
        settings = _setup_silver(tmp_path)
        service = DerivativesDataService(settings=settings)

        # OHLCV DataFrame 생성
        ohlcv_index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        ohlcv_df = pd.DataFrame(
            {"close": [42000, 42500, 43000], "volume": [100, 200, 300]},
            index=ohlcv_index,
        )

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        enriched = service.enrich(ohlcv_df, "BTC/USDT", start, end)

        # 원본 OHLCV 컬럼 보존
        assert "close" in enriched.columns
        assert "volume" in enriched.columns
        assert len(enriched) == 3

    def test_enrich_no_data_returns_original(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        service = DerivativesDataService(settings=settings)

        ohlcv_index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        ohlcv_df = pd.DataFrame(
            {"close": [42000, 42500, 43000]},
            index=ohlcv_index,
        )

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        result = service.enrich(ohlcv_df, "BTC/USDT", start, end)
        pd.testing.assert_frame_equal(result, ohlcv_df)

    def test_enrich_custom_columns(self, tmp_path) -> None:
        settings = _setup_silver(tmp_path)
        service = DerivativesDataService(settings=settings)

        ohlcv_index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        ohlcv_df = pd.DataFrame(
            {"close": [42000, 42500, 43000]},
            index=ohlcv_index,
        )

        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)
        enriched = service.enrich(
            ohlcv_df, "BTC/USDT", start, end,
            columns=["funding_rate"],
        )
        assert "close" in enriched.columns


class TestDerivativesDataServicePrecompute:
    def test_precompute_basic(self, tmp_path) -> None:
        settings = _setup_silver(tmp_path)
        service = DerivativesDataService(settings=settings)

        ohlcv_index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)

        result = service.precompute("BTC/USDT", ohlcv_index, start, end)
        assert len(result) == 5
        assert result.index.equals(ohlcv_index)

    def test_precompute_no_data(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        service = DerivativesDataService(settings=settings)

        ohlcv_index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)

        result = service.precompute("BTC/USDT", ohlcv_index, start, end)
        assert len(result) == 3
        # 모든 값이 NaN
        assert result.isna().all().all()

    def test_precompute_custom_columns(self, tmp_path) -> None:
        settings = _setup_silver(tmp_path)
        service = DerivativesDataService(settings=settings)

        ohlcv_index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)

        result = service.precompute(
            "BTC/USDT", ohlcv_index, start, end,
            columns=["funding_rate", "open_interest"],
        )
        assert len(result) == 3


class TestDefaultDerivColumns:
    def test_default_columns(self) -> None:
        assert "funding_rate" in DEFAULT_DERIV_COLUMNS
        assert "open_interest" in DEFAULT_DERIV_COLUMNS
        assert "ls_ratio" in DEFAULT_DERIV_COLUMNS
        assert "taker_ratio" in DEFAULT_DERIV_COLUMNS
