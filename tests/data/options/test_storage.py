"""Tests for Deribit Options storage (OptionsBronzeStorage, OptionsSilverProcessor)."""

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.data.options.storage import OptionsBronzeStorage, OptionsSilverProcessor


@pytest.fixture
def settings(tmp_path: Path) -> IngestionSettings:
    """테스트용 설정 (tmp_path 사용)."""
    return IngestionSettings(
        options_bronze_dir=tmp_path / "bronze" / "options",
        options_silver_dir=tmp_path / "silver" / "options",
    )


@pytest.fixture
def dvol_df() -> pd.DataFrame:
    """DVOL 테스트 DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"], utc=True),
            "currency": ["BTC"] * 3,
            "open": [60.0, 62.0, 61.0],
            "high": [65.0, 67.0, 66.0],
            "low": [58.0, 60.0, 59.0],
            "close": [63.0, 65.0, 64.0],
            "volume": [100.0, 110.0, 105.0],
            "source": ["deribit"] * 3,
        }
    )


@pytest.fixture
def pc_ratio_df() -> pd.DataFrame:
    """Put/Call Ratio 테스트 DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15"], utc=True),
            "currency": ["BTC"],
            "put_oi": [5000.0],
            "call_oi": [8000.0],
            "pc_ratio": [0.625],
            "source": ["deribit"],
        }
    )


class TestOptionsBronzeStorage:
    """OptionsBronzeStorage 테스트."""

    def test_save_and_load(self, settings: IngestionSettings, dvol_df: pd.DataFrame) -> None:
        """저장 후 로드."""
        storage = OptionsBronzeStorage(settings)
        path = storage.save(dvol_df, "deribit", "btc_dvol")

        assert path.exists()
        loaded = storage.load("deribit", "btc_dvol")
        assert len(loaded) == 3

    def test_save_empty_raises(self, settings: IngestionSettings) -> None:
        """빈 DataFrame 저장 시 ValueError."""
        storage = OptionsBronzeStorage(settings)
        with pytest.raises(ValueError, match="empty"):
            storage.save(pd.DataFrame(), "deribit", "btc_dvol")

    def test_exists(self, settings: IngestionSettings, dvol_df: pd.DataFrame) -> None:
        """파일 존재 확인."""
        storage = OptionsBronzeStorage(settings)
        assert not storage.exists("deribit", "btc_dvol")
        storage.save(dvol_df, "deribit", "btc_dvol")
        assert storage.exists("deribit", "btc_dvol")

    def test_append_dedup(self, settings: IngestionSettings, dvol_df: pd.DataFrame) -> None:
        """append 중복 제거."""
        storage = OptionsBronzeStorage(settings)
        storage.save(dvol_df, "deribit", "btc_dvol")

        new_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
                "currency": ["BTC"] * 2,
                "open": [63.0, 64.0],
                "high": [68.0, 69.0],
                "low": [61.0, 62.0],
                "close": [66.0, 67.0],
                "volume": [115.0, 120.0],
                "source": ["deribit"] * 2,
            }
        )
        storage.append(new_df, "deribit", "btc_dvol")

        loaded = storage.load("deribit", "btc_dvol")
        assert len(loaded) == 4  # 3 + 2 - 1 overlap

    def test_append_creates_if_not_exists(
        self, settings: IngestionSettings, pc_ratio_df: pd.DataFrame
    ) -> None:
        """append: 파일 없으면 save 호출."""
        storage = OptionsBronzeStorage(settings)
        path = storage.append(pc_ratio_df, "deribit", "btc_pc_ratio")
        assert path.exists()
        loaded = storage.load("deribit", "btc_pc_ratio")
        assert len(loaded) == 1

    def test_get_info(self, settings: IngestionSettings, dvol_df: pd.DataFrame) -> None:
        """파일 정보 조회."""
        storage = OptionsBronzeStorage(settings)
        assert storage.get_info("deribit", "btc_dvol") is None
        storage.save(dvol_df, "deribit", "btc_dvol")
        info = storage.get_info("deribit", "btc_dvol")
        assert info is not None
        assert "size_bytes" in info


class TestOptionsSilverProcessor:
    """OptionsSilverProcessor 테스트."""

    def test_process_dvol(self, settings: IngestionSettings, dvol_df: pd.DataFrame) -> None:
        """DVOL Bronze → Silver 변환."""
        bronze = OptionsBronzeStorage(settings)
        bronze.save(dvol_df, "deribit", "btc_dvol")

        silver = OptionsSilverProcessor(settings, bronze)
        path = silver.process("deribit", "btc_dvol")

        assert path.exists()
        loaded = silver.load("deribit", "btc_dvol")
        assert len(loaded) == 3
        # close 컬럼이 numeric으로 coerced
        assert loaded["close"].dtype in ("float64", "Float64")

    def test_process_pc_ratio(self, settings: IngestionSettings, pc_ratio_df: pd.DataFrame) -> None:
        """P/C Ratio Bronze → Silver 변환."""
        bronze = OptionsBronzeStorage(settings)
        bronze.save(pc_ratio_df, "deribit", "btc_pc_ratio")

        silver = OptionsSilverProcessor(settings, bronze)
        path = silver.process("deribit", "btc_pc_ratio")

        assert path.exists()
        loaded = silver.load("deribit", "btc_pc_ratio")
        assert len(loaded) == 1

    def test_process_dedup(self, settings: IngestionSettings) -> None:
        """중복 제거 검증."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-01-16"], utc=True),
                "currency": ["BTC"] * 3,
                "close": [63.0, 64.0, 65.0],
                "source": ["deribit"] * 3,
            }
        )

        bronze = OptionsBronzeStorage(settings)
        bronze.save(df, "deribit", "btc_dvol")

        silver = OptionsSilverProcessor(settings, bronze)
        silver.process("deribit", "btc_dvol")

        loaded = silver.load("deribit", "btc_dvol")
        assert len(loaded) == 2  # 중복 제거

    def test_exists(self, settings: IngestionSettings, dvol_df: pd.DataFrame) -> None:
        """Silver 파일 존재 확인."""
        bronze = OptionsBronzeStorage(settings)
        bronze.save(dvol_df, "deribit", "btc_dvol")

        silver = OptionsSilverProcessor(settings, bronze)
        assert not silver.exists("deribit", "btc_dvol")

        silver.process("deribit", "btc_dvol")
        assert silver.exists("deribit", "btc_dvol")

    def test_text_cols_preserved(self, settings: IngestionSettings) -> None:
        """텍스트 컬럼은 numeric coercion에서 제외."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-15"], utc=True),
                "currency": ["BTC"],
                "near_expiry": ["BTC-26JAN24"],
                "far_expiry": ["BTC-29MAR24"],
                "near_basis_pct": [1.2],
                "far_basis_pct": [3.5],
                "slope": [2.3],
                "source": ["deribit"],
            }
        )

        bronze = OptionsBronzeStorage(settings)
        bronze.save(df, "deribit", "btc_term_structure")

        silver = OptionsSilverProcessor(settings, bronze)
        silver.process("deribit", "btc_term_structure")

        loaded = silver.load("deribit", "btc_term_structure")
        assert loaded.iloc[0]["near_expiry"] == "BTC-26JAN24"
        assert loaded.iloc[0]["currency"] == "BTC"
