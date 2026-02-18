"""Tests for macro data storage (MacroBronzeStorage, MacroSilverProcessor)."""

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.data.macro.storage import MacroBronzeStorage, MacroSilverProcessor


@pytest.fixture
def settings(tmp_path: Path) -> IngestionSettings:
    """테스트용 설정 (tmp_path 사용)."""
    return IngestionSettings(
        macro_bronze_dir=tmp_path / "bronze" / "macro",
        macro_silver_dir=tmp_path / "silver" / "macro",
    )


@pytest.fixture
def fred_df() -> pd.DataFrame:
    """FRED 테스트 DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"], utc=True),
            "value": [123.45, None, 124.00],
            "series_id": ["DTWEXBGS"] * 3,
            "source": ["fred"] * 3,
        }
    )


@pytest.fixture
def yfinance_df() -> pd.DataFrame:
    """yfinance 테스트 DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
            "open": [470.0, 471.0],
            "high": [475.0, 476.0],
            "low": [468.0, 469.0],
            "close": [473.5, 474.0],
            "volume": [50000000, 51000000],
            "ticker": ["SPY", "SPY"],
            "source": ["yfinance", "yfinance"],
        }
    )


class TestMacroBronzeStorage:
    """MacroBronzeStorage 테스트."""

    def test_save_and_load(self, settings: IngestionSettings, fred_df: pd.DataFrame) -> None:
        """저장 후 로드."""
        storage = MacroBronzeStorage(settings)
        path = storage.save(fred_df, "fred", "dxy")

        assert path.exists()
        loaded = storage.load("fred", "dxy")
        assert len(loaded) == 3

    def test_save_empty_raises(self, settings: IngestionSettings) -> None:
        """빈 DataFrame 저장 시 ValueError."""
        storage = MacroBronzeStorage(settings)
        with pytest.raises(ValueError, match="empty"):
            storage.save(pd.DataFrame(), "fred", "dxy")

    def test_exists(self, settings: IngestionSettings, fred_df: pd.DataFrame) -> None:
        """파일 존재 확인."""
        storage = MacroBronzeStorage(settings)
        assert not storage.exists("fred", "dxy")
        storage.save(fred_df, "fred", "dxy")
        assert storage.exists("fred", "dxy")

    def test_append_dedup(self, settings: IngestionSettings, fred_df: pd.DataFrame) -> None:
        """append 중복 제거."""
        storage = MacroBronzeStorage(settings)
        storage.save(fred_df, "fred", "dxy")

        new_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-17", "2024-01-18"], utc=True),
                "value": [125.0, 126.0],
                "series_id": ["DTWEXBGS"] * 2,
                "source": ["fred"] * 2,
            }
        )
        storage.append(new_df, "fred", "dxy")

        loaded = storage.load("fred", "dxy")
        assert len(loaded) == 4  # 3 + 2 - 1 overlap

    def test_get_info(self, settings: IngestionSettings, fred_df: pd.DataFrame) -> None:
        """파일 정보 조회."""
        storage = MacroBronzeStorage(settings)
        assert storage.get_info("fred", "dxy") is None
        storage.save(fred_df, "fred", "dxy")
        info = storage.get_info("fred", "dxy")
        assert info is not None
        assert "size_bytes" in info


class TestMacroSilverProcessor:
    """MacroSilverProcessor 테스트."""

    def test_process_fred(self, settings: IngestionSettings, fred_df: pd.DataFrame) -> None:
        """FRED Bronze → Silver 변환."""
        bronze = MacroBronzeStorage(settings)
        bronze.save(fred_df, "fred", "dxy")

        silver = MacroSilverProcessor(settings, bronze)
        path = silver.process("fred", "dxy")

        assert path.exists()
        loaded = silver.load("fred", "dxy")
        assert len(loaded) == 3
        # value 컬럼이 numeric으로 coerced
        assert loaded["value"].dtype in ("float64", "Float64")

    def test_process_yfinance(self, settings: IngestionSettings, yfinance_df: pd.DataFrame) -> None:
        """yfinance Bronze → Silver 변환."""
        bronze = MacroBronzeStorage(settings)
        bronze.save(yfinance_df, "yfinance", "spy")

        silver = MacroSilverProcessor(settings, bronze)
        path = silver.process("yfinance", "spy")

        assert path.exists()
        loaded = silver.load("yfinance", "spy")
        assert len(loaded) == 2

    def test_process_dedup(self, settings: IngestionSettings) -> None:
        """중복 제거 검증."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-15", "2024-01-15", "2024-01-16"], utc=True),
                "value": [123.0, 124.0, 125.0],
                "series_id": ["DGS10"] * 3,
                "source": ["fred"] * 3,
            }
        )

        bronze = MacroBronzeStorage(settings)
        bronze.save(df, "fred", "dgs10")

        silver = MacroSilverProcessor(settings, bronze)
        silver.process("fred", "dgs10")

        loaded = silver.load("fred", "dgs10")
        assert len(loaded) == 2  # 중복 제거

    def test_exists(self, settings: IngestionSettings, fred_df: pd.DataFrame) -> None:
        """Silver 파일 존재 확인."""
        bronze = MacroBronzeStorage(settings)
        bronze.save(fred_df, "fred", "dxy")

        silver = MacroSilverProcessor(settings, bronze)
        assert not silver.exists("fred", "dxy")

        silver.process("fred", "dxy")
        assert silver.exists("fred", "dxy")
