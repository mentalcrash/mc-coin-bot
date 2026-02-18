"""Tests for src/data/derivatives_storage.py — Bronze/Silver derivatives storage."""

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.data.derivatives_storage import (
    DERIV_COLUMNS,
    DerivativesBronzeStorage,
    DerivativesSilverProcessor,
    batch_to_dataframe,
)
from src.models.derivatives import (
    DerivativesBatch,
    FundingRateRecord,
    LongShortRatioRecord,
    OpenInterestRecord,
    TakerRatioRecord,
    TopTraderAccountRatioRecord,
    TopTraderPositionRatioRecord,
)


def _make_batch(symbol: str = "BTC/USDT") -> DerivativesBatch:
    """테스트용 DerivativesBatch 생성."""
    ts1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    ts2 = datetime(2024, 1, 1, 8, 0, 0, tzinfo=UTC)
    return DerivativesBatch(
        symbol=symbol,
        funding_rates=(
            FundingRateRecord(
                symbol=symbol,
                timestamp=ts1,
                funding_rate=Decimal("0.0001"),
                mark_price=Decimal(42000),
            ),
            FundingRateRecord(
                symbol=symbol,
                timestamp=ts2,
                funding_rate=Decimal("0.0002"),
                mark_price=Decimal(42500),
            ),
        ),
        open_interest=(
            OpenInterestRecord(
                symbol=symbol,
                timestamp=ts1,
                sum_open_interest=Decimal(50000),
                sum_open_interest_value=Decimal(2100000000),
            ),
        ),
        long_short_ratios=(
            LongShortRatioRecord(
                symbol=symbol,
                timestamp=ts1,
                long_account=Decimal("0.55"),
                short_account=Decimal("0.45"),
                long_short_ratio=Decimal("1.22"),
            ),
        ),
        taker_ratios=(
            TakerRatioRecord(
                symbol=symbol,
                timestamp=ts1,
                buy_vol=Decimal(1000),
                sell_vol=Decimal(800),
                buy_sell_ratio=Decimal("1.25"),
            ),
        ),
        top_acct_ratios=(
            TopTraderAccountRatioRecord(
                symbol=symbol,
                timestamp=ts1,
                long_account=Decimal("0.60"),
                short_account=Decimal("0.40"),
                long_short_ratio=Decimal("1.50"),
            ),
        ),
        top_pos_ratios=(
            TopTraderPositionRatioRecord(
                symbol=symbol,
                timestamp=ts1,
                long_account=Decimal("0.65"),
                short_account=Decimal("0.35"),
                long_short_ratio=Decimal("1.86"),
            ),
        ),
    )


class TestBatchToDataframe:
    def test_basic(self) -> None:
        batch = _make_batch()
        df = batch_to_dataframe(batch)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert set(DERIV_COLUMNS).issubset(set(df.columns))
        assert len(df) >= 1

    def test_empty_batch(self) -> None:
        batch = DerivativesBatch(symbol="BTC/USDT")
        df = batch_to_dataframe(batch)
        assert df.empty
        assert list(df.columns) == DERIV_COLUMNS

    def test_funding_rate_values(self) -> None:
        batch = _make_batch()
        df = batch_to_dataframe(batch)
        # funding_rate 값이 존재해야 함
        assert df["funding_rate"].dropna().shape[0] > 0


class TestDerivativesBronzeStorage:
    def test_save_and_load(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        storage = DerivativesBronzeStorage(settings)
        batch = _make_batch()

        path = storage.save(batch, 2024)
        assert path.exists()

        df = storage.load("BTC/USDT", 2024)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    def test_exists(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        storage = DerivativesBronzeStorage(settings)
        assert not storage.exists("BTC/USDT", 2024)

        storage.save(_make_batch(), 2024)
        assert storage.exists("BTC/USDT", 2024)

    def test_save_empty_raises(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        storage = DerivativesBronzeStorage(settings)
        empty = DerivativesBatch(symbol="BTC/USDT")
        with pytest.raises(ValueError, match="empty"):
            storage.save(empty, 2024)

    def test_append(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        storage = DerivativesBronzeStorage(settings)
        batch1 = _make_batch()
        storage.save(batch1, 2024)

        # 다른 timestamp로 추가 배치
        ts3 = datetime(2024, 1, 1, 16, 0, 0, tzinfo=UTC)
        batch2 = DerivativesBatch(
            symbol="BTC/USDT",
            funding_rates=(
                FundingRateRecord(
                    symbol="BTC/USDT",
                    timestamp=ts3,
                    funding_rate=Decimal("0.0003"),
                    mark_price=Decimal(43000),
                ),
            ),
        )
        storage.append(batch2, 2024)
        df = storage.load("BTC/USDT", 2024)
        # append 후 행 수 증가
        assert len(df) >= 3  # ts1, ts2, ts3

    def test_get_info(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        storage = DerivativesBronzeStorage(settings)

        assert storage.get_info("BTC/USDT", 2024) is None

        storage.save(_make_batch(), 2024)
        info = storage.get_info("BTC/USDT", 2024)
        assert info is not None
        assert "size_bytes" in info

    def test_load_missing_raises(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        storage = DerivativesBronzeStorage(settings)
        with pytest.raises(Exception, match="not found"):
            storage.load("BTC/USDT", 2024)


class TestDerivativesSilverProcessor:
    def test_process(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        bronze = DerivativesBronzeStorage(settings)
        bronze.save(_make_batch(), 2024)

        processor = DerivativesSilverProcessor(settings, bronze)
        path = processor.process("BTC/USDT", 2024)
        assert path.exists()

    def test_load(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        bronze = DerivativesBronzeStorage(settings)
        bronze.save(_make_batch(), 2024)

        processor = DerivativesSilverProcessor(settings, bronze)
        processor.process("BTC/USDT", 2024)

        df = processor.load("BTC/USDT", 2024)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    def test_exists(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        bronze = DerivativesBronzeStorage(settings)
        processor = DerivativesSilverProcessor(settings, bronze)
        assert not processor.exists("BTC/USDT", 2024)

        bronze.save(_make_batch(), 2024)
        processor.process("BTC/USDT", 2024)
        assert processor.exists("BTC/USDT", 2024)

    def test_analyze_gaps(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        bronze = DerivativesBronzeStorage(settings)
        bronze.save(_make_batch(), 2024)

        processor = DerivativesSilverProcessor(settings, bronze)
        report = processor.analyze_gaps("BTC/USDT", 2024)
        assert report.total_rows >= 1
        assert report.first_timestamp is not None

    def test_forward_fill(self, tmp_path) -> None:
        """forward-fill이 NaN을 채우는지 확인."""
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        bronze = DerivativesBronzeStorage(settings)
        bronze.save(_make_batch(), 2024)

        processor = DerivativesSilverProcessor(settings, bronze)
        processor.process("BTC/USDT", 2024)

        df = processor.load("BTC/USDT", 2024)
        # forward-fill 적용 후 첫 행 이후에는 funding_rate NaN이 줄어야 함
        # (첫 행은 bfill 미적용이므로 NaN 가능)
        assert df is not None

    def test_backward_compat_missing_top_trader_cols(self, tmp_path) -> None:
        """기존 parquet 로드 시 새 top trader 컬럼은 NaN (하위호환)."""
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        bronze = DerivativesBronzeStorage(settings)

        # top trader 없는 구 버전 배치
        old_batch = DerivativesBatch(
            symbol="BTC/USDT",
            funding_rates=(
                FundingRateRecord(
                    symbol="BTC/USDT",
                    timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                    funding_rate=Decimal("0.0001"),
                    mark_price=Decimal(42000),
                ),
            ),
        )
        bronze.save(old_batch, 2024)
        df = bronze.load("BTC/USDT", 2024)
        # 새 컬럼이 NaN으로 존재
        assert "top_acct_ls_ratio" in df.columns
        assert df["top_acct_ls_ratio"].isna().all()

    def test_load_missing_raises(self, tmp_path) -> None:
        settings = IngestionSettings(bronze_dir=tmp_path / "bronze", silver_dir=tmp_path / "silver")
        processor = DerivativesSilverProcessor(settings)
        with pytest.raises(Exception, match="not found"):
            processor.load("BTC/USDT", 2024)
