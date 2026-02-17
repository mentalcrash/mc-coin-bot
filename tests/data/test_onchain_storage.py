"""Tests for src/data/onchain/storage.py — Bronze/Silver on-chain storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.core.exceptions import StorageError
from src.data.onchain.storage import OnchainBronzeStorage, OnchainSilverProcessor


def _make_stablecoin_df(n: int = 5) -> pd.DataFrame:
    """테스트용 stablecoin DataFrame 생성."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "date": dates,
        "total_circulating_usd": [100_000_000_000 + i * 1_000_000 for i in range(n)],
        "source": ["defillama"] * n,
    })


def _make_settings(tmp_path: Path) -> IngestionSettings:
    """테스트용 IngestionSettings."""
    return IngestionSettings(
        onchain_bronze_dir=tmp_path / "bronze" / "onchain",
        onchain_silver_dir=tmp_path / "silver" / "onchain",
    )


class TestOnchainBronzeStorage:
    def test_save_and_load(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        storage = OnchainBronzeStorage(settings)
        df = _make_stablecoin_df()

        path = storage.save(df, "defillama", "stablecoin_total")
        assert path.exists()

        loaded = storage.load("defillama", "stablecoin_total")
        assert len(loaded) == 5

    def test_save_empty_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        storage = OnchainBronzeStorage(settings)

        with pytest.raises(ValueError, match="empty"):
            storage.save(pd.DataFrame(), "defillama", "stablecoin_total")

    def test_exists(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        storage = OnchainBronzeStorage(settings)

        assert not storage.exists("defillama", "stablecoin_total")

        storage.save(_make_stablecoin_df(), "defillama", "stablecoin_total")
        assert storage.exists("defillama", "stablecoin_total")

    def test_append_dedup(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        storage = OnchainBronzeStorage(settings)

        # Save initial data
        df1 = _make_stablecoin_df(3)
        storage.save(df1, "defillama", "stablecoin_total")

        # Append with overlap (last 2 of df1 + 2 new)
        dates = pd.date_range("2024-01-02", periods=4, freq="D", tz="UTC")
        df2 = pd.DataFrame({
            "date": dates,
            "total_circulating_usd": [200_000_000_000 + i * 1_000_000 for i in range(4)],
            "source": ["defillama"] * 4,
        })
        storage.append(df2, "defillama", "stablecoin_total", dedup_col="date")

        loaded = storage.load("defillama", "stablecoin_total")
        # 3 original + 2 new (2 overlap kept="last")
        assert len(loaded) == 5

    def test_append_creates_if_not_exists(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        storage = OnchainBronzeStorage(settings)

        df = _make_stablecoin_df(2)
        path = storage.append(df, "defillama", "stablecoin_total")
        assert path.exists()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        storage = OnchainBronzeStorage(settings)

        with pytest.raises(StorageError, match="not found"):
            storage.load("defillama", "nonexistent")

    def test_get_info(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        storage = OnchainBronzeStorage(settings)

        assert storage.get_info("defillama", "stablecoin_total") is None

        storage.save(_make_stablecoin_df(), "defillama", "stablecoin_total")
        info = storage.get_info("defillama", "stablecoin_total")
        assert info is not None
        assert "path" in info
        assert "size_bytes" in info
        assert info["size_bytes"] > 0


class TestOnchainSilverProcessor:
    def test_process(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        bronze = OnchainBronzeStorage(settings)
        processor = OnchainSilverProcessor(settings, bronze)

        # Save bronze data first
        df = _make_stablecoin_df()
        bronze.save(df, "defillama", "stablecoin_total")

        # Process to silver
        path = processor.process("defillama", "stablecoin_total")
        assert path.exists()

        loaded = processor.load("defillama", "stablecoin_total")
        assert len(loaded) == 5

    def test_process_dedup(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        bronze = OnchainBronzeStorage(settings)
        processor = OnchainSilverProcessor(settings, bronze)

        # Create df with duplicate dates
        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        dup_dates = [*list(dates), dates[1]]  # duplicate Jan 2
        df = pd.DataFrame({
            "date": dup_dates,
            "total_circulating_usd": [100, 200, 300, 250],
            "source": ["defillama"] * 4,
        })
        bronze.save(df, "defillama", "stablecoin_dedup")

        path = processor.process("defillama", "stablecoin_dedup")
        loaded = processor.load("defillama", "stablecoin_dedup")
        assert len(loaded) == 3  # dedup removed 1
        assert path.exists()

    def test_process_sort(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        bronze = OnchainBronzeStorage(settings)
        processor = OnchainSilverProcessor(settings, bronze)

        # Create unsorted data
        dates = [
            pd.Timestamp("2024-01-03", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        ]
        df = pd.DataFrame({
            "date": dates,
            "total_circulating_usd": [300, 100, 200],
            "source": ["defillama"] * 3,
        })
        bronze.save(df, "defillama", "stablecoin_sort")

        processor.process("defillama", "stablecoin_sort")
        loaded = processor.load("defillama", "stablecoin_sort")

        # Check sorted
        date_vals = loaded["date"].tolist()
        assert date_vals == sorted(date_vals)

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        processor = OnchainSilverProcessor(settings)

        with pytest.raises(StorageError, match="not found"):
            processor.load("defillama", "nonexistent")

    def test_missing_bronze_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        processor = OnchainSilverProcessor(settings)

        with pytest.raises(StorageError, match="not found"):
            processor.process("defillama", "nonexistent")

    def test_exists(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        bronze = OnchainBronzeStorage(settings)
        processor = OnchainSilverProcessor(settings, bronze)

        assert not processor.exists("defillama", "stablecoin_total")

        bronze.save(_make_stablecoin_df(), "defillama", "stablecoin_total")
        processor.process("defillama", "stablecoin_total")
        assert processor.exists("defillama", "stablecoin_total")

    def test_silver_converts_decimal_to_float64(self, tmp_path: Path) -> None:
        """Decimal/object 컬럼이 float64로 변환되는지 검증."""
        from decimal import Decimal

        settings = _make_settings(tmp_path)
        bronze = OnchainBronzeStorage(settings)
        processor = OnchainSilverProcessor(settings, bronze)

        dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        df = pd.DataFrame({
            "date": dates,
            "value": [Decimal(72), Decimal(68), Decimal(75)],
            "source": ["test"] * 3,
        })
        bronze.save(df, "test_src", "decimal_test")
        processor.process("test_src", "decimal_test")

        loaded = processor.load("test_src", "decimal_test")
        # Decimal → float64 변환 확인
        assert loaded["value"].dtype in ("float64", "int64")
        assert loaded["value"].iloc[0] == 72.0
        # source 컬럼은 text 유지
        assert loaded["source"].dtype == object
