"""Tests for MacroDataService."""

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.data.macro.service import MACRO_BATCH_DEFINITIONS, MacroDataService
from src.data.macro.storage import MacroBronzeStorage, MacroSilverProcessor


@pytest.fixture
def settings(tmp_path: Path) -> IngestionSettings:
    """테스트용 설정."""
    return IngestionSettings(
        macro_bronze_dir=tmp_path / "bronze" / "macro",
        macro_silver_dir=tmp_path / "silver" / "macro",
    )


@pytest.fixture
def service_with_data(settings: IngestionSettings) -> MacroDataService:
    """Silver 데이터가 준비된 서비스."""
    bronze = MacroBronzeStorage(settings)
    silver = MacroSilverProcessor(settings, bronze)

    # FRED DXY 데이터
    fred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
            "value": [123.45, 124.00],
            "series_id": ["DTWEXBGS"] * 2,
            "source": ["fred"] * 2,
        }
    )
    bronze.save(fred_df, "fred", "dxy")
    silver.process("fred", "dxy")

    # yfinance SPY 데이터
    yf_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
            "open": [470.0, 471.0],
            "high": [475.0, 476.0],
            "low": [468.0, 469.0],
            "close": [473.5, 474.0],
            "volume": [50000000, 51000000],
            "ticker": ["SPY"] * 2,
            "source": ["yfinance"] * 2,
        }
    )
    bronze.save(yf_df, "yfinance", "spy")
    silver.process("yfinance", "spy")

    return MacroDataService(settings, silver, catalog=None)


class TestMacroDataService:
    """MacroDataService 테스트."""

    def test_batch_definitions_fred(self, settings: IngestionSettings) -> None:
        """FRED batch definitions."""
        service = MacroDataService(settings, catalog=None)
        defs = service.get_batch_definitions("fred")
        assert len(defs) == 6  # gold removed (FRED series discontinued 2022-01)
        assert all(s == "fred" for s, _ in defs)

    def test_batch_definitions_yfinance(self, settings: IngestionSettings) -> None:
        """yfinance batch definitions."""
        service = MacroDataService(settings, catalog=None)
        defs = service.get_batch_definitions("yfinance")
        assert len(defs) == 6
        assert all(s == "yfinance" for s, _ in defs)

    def test_batch_definitions_all(self, settings: IngestionSettings) -> None:
        """전체 batch definitions."""
        service = MacroDataService(settings, catalog=None)
        defs = service.get_batch_definitions("all")
        assert len(defs) == 14  # gold removed

    def test_batch_definitions_invalid(self, settings: IngestionSettings) -> None:
        """잘못된 batch type."""
        service = MacroDataService(settings, catalog=None)
        with pytest.raises(ValueError, match="Unknown batch type"):
            service.get_batch_definitions("invalid")

    def test_load(self, service_with_data: MacroDataService) -> None:
        """Silver 로드."""
        df = service_with_data.load("fred", "dxy")
        assert len(df) == 2

    def test_enrich(self, service_with_data: MacroDataService) -> None:
        """OHLCV에 macro 데이터 병합."""
        ohlcv = pd.DataFrame(
            {"close": [50000.0, 51000.0]},
            index=pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
        )

        enriched = service_with_data.enrich(ohlcv, "fred", "dxy", columns=["value"], lag_days=0)
        assert "value" in enriched.columns
        assert len(enriched) == 2

    def test_enrich_with_lag(self, service_with_data: MacroDataService) -> None:
        """Lag 적용 enrichment."""
        ohlcv = pd.DataFrame(
            {"close": [50000.0, 51000.0, 52000.0]},
            index=pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"], utc=True),
        )

        # lag=1일이면 1/15 데이터는 1/16부터 사용 가능
        enriched = service_with_data.enrich(ohlcv, "fred", "dxy", columns=["value"], lag_days=1)
        assert pd.isna(enriched.iloc[0]["value"])  # 1/15에는 아직 데이터 없음

    def test_batch_definitions_constant(self) -> None:
        """MACRO_BATCH_DEFINITIONS 상수 확인."""
        assert "fred" in MACRO_BATCH_DEFINITIONS
        assert "yfinance" in MACRO_BATCH_DEFINITIONS
        assert len(MACRO_BATCH_DEFINITIONS["fred"]) == 6  # gold removed
        assert len(MACRO_BATCH_DEFINITIONS["yfinance"]) == 6
