"""Tests for Extended Derivatives storage (DerivExtBronzeStorage, DerivExtSilverProcessor)."""

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.data.deriv_ext.storage import DerivExtBronzeStorage, DerivExtSilverProcessor


@pytest.fixture
def settings(tmp_path: Path) -> IngestionSettings:
    """테스트용 설정 (tmp_path 사용)."""
    return IngestionSettings(
        deriv_ext_bronze_dir=tmp_path / "bronze" / "deriv_ext",
        deriv_ext_silver_dir=tmp_path / "silver" / "deriv_ext",
    )


@pytest.fixture
def oi_df() -> pd.DataFrame:
    """Aggregated OI 테스트 DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"], utc=True),
            "symbol": ["BTCUSDT.6"] * 3,
            "open": [50000.0, 51000.0, 52000.0],
            "high": [52000.0, 53000.0, 54000.0],
            "low": [49000.0, 50000.0, 51000.0],
            "close": [51000.0, 52000.0, 53000.0],
            "source": ["coinalyze"] * 3,
        }
    )


@pytest.fixture
def liquidation_hourly_df() -> pd.DataFrame:
    """Liquidation hourly 테스트 DataFrame (Silver에서 daily 롤업 대상)."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-15 00:00",
                    "2024-01-15 01:00",
                    "2024-01-15 02:00",
                    "2024-01-16 00:00",
                ],
                utc=True,
            ),
            "symbol": ["BTCUSDT.6"] * 4,
            "long_volume": [100000.0, 50000.0, 30000.0, 200000.0],
            "short_volume": [80000.0, 40000.0, 20000.0, 150000.0],
            "source": ["coinalyze"] * 4,
        }
    )


class TestDerivExtBronzeStorage:
    """DerivExtBronzeStorage 테스트."""

    def test_save_and_load(self, settings: IngestionSettings, oi_df: pd.DataFrame) -> None:
        """저장 후 로드."""
        storage = DerivExtBronzeStorage(settings)
        path = storage.save(oi_df, "coinalyze", "btc_agg_oi")

        assert path.exists()
        loaded = storage.load("coinalyze", "btc_agg_oi")
        assert len(loaded) == 3

    def test_save_empty_raises(self, settings: IngestionSettings) -> None:
        """빈 DataFrame 저장 시 ValueError."""
        storage = DerivExtBronzeStorage(settings)
        with pytest.raises(ValueError, match="empty"):
            storage.save(pd.DataFrame(), "coinalyze", "btc_agg_oi")

    def test_exists(self, settings: IngestionSettings, oi_df: pd.DataFrame) -> None:
        """파일 존재 확인."""
        storage = DerivExtBronzeStorage(settings)
        assert not storage.exists("coinalyze", "btc_agg_oi")
        storage.save(oi_df, "coinalyze", "btc_agg_oi")
        assert storage.exists("coinalyze", "btc_agg_oi")

    def test_append_dedup(self, settings: IngestionSettings, oi_df: pd.DataFrame) -> None:
        """append 중복 제거."""
        storage = DerivExtBronzeStorage(settings)
        storage.save(oi_df, "coinalyze", "btc_agg_oi")

        new_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
                "symbol": ["BTCUSDT.6"] * 2,
                "open": [53000.0, 54000.0],
                "high": [55000.0, 56000.0],
                "low": [52000.0, 53000.0],
                "close": [54000.0, 55000.0],
                "source": ["coinalyze"] * 2,
            }
        )
        storage.append(new_df, "coinalyze", "btc_agg_oi")

        loaded = storage.load("coinalyze", "btc_agg_oi")
        assert len(loaded) == 4  # 3 + 2 - 1 overlap

    def test_append_creates_if_not_exists(
        self, settings: IngestionSettings, oi_df: pd.DataFrame
    ) -> None:
        """append: 파일 없으면 save 호출."""
        storage = DerivExtBronzeStorage(settings)
        path = storage.append(oi_df, "coinalyze", "btc_agg_oi")
        assert path.exists()
        loaded = storage.load("coinalyze", "btc_agg_oi")
        assert len(loaded) == 3

    def test_get_info(self, settings: IngestionSettings, oi_df: pd.DataFrame) -> None:
        """파일 정보 조회."""
        storage = DerivExtBronzeStorage(settings)
        assert storage.get_info("coinalyze", "btc_agg_oi") is None
        storage.save(oi_df, "coinalyze", "btc_agg_oi")
        info = storage.get_info("coinalyze", "btc_agg_oi")
        assert info is not None
        assert "size_bytes" in info


class TestDerivExtSilverProcessor:
    """DerivExtSilverProcessor 테스트."""

    def test_process_oi(self, settings: IngestionSettings, oi_df: pd.DataFrame) -> None:
        """OI Bronze → Silver 변환."""
        bronze = DerivExtBronzeStorage(settings)
        bronze.save(oi_df, "coinalyze", "btc_agg_oi")

        silver = DerivExtSilverProcessor(settings, bronze)
        path = silver.process("coinalyze", "btc_agg_oi")

        assert path.exists()
        loaded = silver.load("coinalyze", "btc_agg_oi")
        assert len(loaded) == 3
        # close 컬럼이 numeric으로 coerced
        assert loaded["close"].dtype in ("float64", "Float64")

    def test_process_dedup(self, settings: IngestionSettings) -> None:
        """중복 제거 검증."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-01-16"], utc=True),
                "symbol": ["BTCUSDT.6"] * 3,
                "close": [51000.0, 52000.0, 53000.0],
                "source": ["coinalyze"] * 3,
            }
        )

        bronze = DerivExtBronzeStorage(settings)
        bronze.save(df, "coinalyze", "btc_agg_oi")

        silver = DerivExtSilverProcessor(settings, bronze)
        silver.process("coinalyze", "btc_agg_oi")

        loaded = silver.load("coinalyze", "btc_agg_oi")
        assert len(loaded) == 2  # 중복 제거

    def test_liquidation_rollup(
        self, settings: IngestionSettings, liquidation_hourly_df: pd.DataFrame
    ) -> None:
        """Liquidation hourly → daily 롤업."""
        bronze = DerivExtBronzeStorage(settings)
        bronze.save(liquidation_hourly_df, "coinalyze", "btc_liquidations")

        silver = DerivExtSilverProcessor(settings, bronze)
        silver.process("coinalyze", "btc_liquidations")

        loaded = silver.load("coinalyze", "btc_liquidations")
        assert len(loaded) == 2  # 2일치로 롤업

        # 2024-01-15: 100k + 50k + 30k = 180k long
        day1 = loaded[loaded["date"].dt.day == 15]
        assert len(day1) == 1
        assert day1.iloc[0]["long_volume"] == pytest.approx(180000.0)
        assert day1.iloc[0]["short_volume"] == pytest.approx(140000.0)

    def test_exists(self, settings: IngestionSettings, oi_df: pd.DataFrame) -> None:
        """Silver 파일 존재 확인."""
        bronze = DerivExtBronzeStorage(settings)
        bronze.save(oi_df, "coinalyze", "btc_agg_oi")

        silver = DerivExtSilverProcessor(settings, bronze)
        assert not silver.exists("coinalyze", "btc_agg_oi")

        silver.process("coinalyze", "btc_agg_oi")
        assert silver.exists("coinalyze", "btc_agg_oi")

    def test_text_cols_preserved(self, settings: IngestionSettings) -> None:
        """텍스트 컬럼은 numeric coercion에서 제외."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-15"], utc=True),
                "symbol": ["BTCUSDT.6"],
                "close": [51000.0],
                "source": ["coinalyze"],
            }
        )

        bronze = DerivExtBronzeStorage(settings)
        bronze.save(df, "coinalyze", "btc_agg_oi")

        silver = DerivExtSilverProcessor(settings, bronze)
        silver.process("coinalyze", "btc_agg_oi")

        loaded = silver.load("coinalyze", "btc_agg_oi")
        assert loaded.iloc[0]["symbol"] == "BTCUSDT.6"
        assert loaded.iloc[0]["source"] == "coinalyze"
