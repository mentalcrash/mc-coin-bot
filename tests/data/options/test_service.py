"""Tests for OptionsDataService."""

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.data.options.service import OPTIONS_BATCH_DEFINITIONS, OptionsDataService
from src.data.options.storage import OptionsBronzeStorage, OptionsSilverProcessor


@pytest.fixture
def settings(tmp_path: Path) -> IngestionSettings:
    """테스트용 설정."""
    return IngestionSettings(
        options_bronze_dir=tmp_path / "bronze" / "options",
        options_silver_dir=tmp_path / "silver" / "options",
    )


@pytest.fixture
def service_with_data(settings: IngestionSettings) -> OptionsDataService:
    """Silver 데이터가 준비된 서비스."""
    bronze = OptionsBronzeStorage(settings)
    silver = OptionsSilverProcessor(settings, bronze)

    # DVOL 데이터
    dvol_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
            "currency": ["BTC"] * 2,
            "close": [63.0, 65.0],
            "source": ["deribit"] * 2,
        }
    )
    bronze.save(dvol_df, "deribit", "btc_dvol")
    silver.process("deribit", "btc_dvol")

    # P/C Ratio 데이터
    pcr_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
            "currency": ["BTC"] * 2,
            "put_oi": [5000.0, 5200.0],
            "call_oi": [8000.0, 8100.0],
            "pc_ratio": [0.625, 0.642],
            "source": ["deribit"] * 2,
        }
    )
    bronze.save(pcr_df, "deribit", "btc_pc_ratio")
    silver.process("deribit", "btc_pc_ratio")

    return OptionsDataService(settings, silver, catalog=None)


class TestOptionsDataService:
    """OptionsDataService 테스트."""

    def test_batch_definitions_deribit(self, settings: IngestionSettings) -> None:
        """Deribit batch definitions."""
        service = OptionsDataService(settings, catalog=None)
        defs = service.get_batch_definitions("deribit")
        assert len(defs) == 6
        assert all(s == "deribit" for s, _ in defs)

    def test_batch_definitions_all(self, settings: IngestionSettings) -> None:
        """전체 batch definitions."""
        service = OptionsDataService(settings, catalog=None)
        defs = service.get_batch_definitions("all")
        assert len(defs) == 6

    def test_batch_definitions_invalid(self, settings: IngestionSettings) -> None:
        """잘못된 batch type."""
        service = OptionsDataService(settings, catalog=None)
        with pytest.raises(ValueError, match="Unknown batch type"):
            service.get_batch_definitions("invalid")

    def test_load(self, service_with_data: OptionsDataService) -> None:
        """Silver 로드."""
        df = service_with_data.load("deribit", "btc_dvol")
        assert len(df) == 2

    def test_enrich(self, service_with_data: OptionsDataService) -> None:
        """OHLCV에 options 데이터 병합."""
        ohlcv = pd.DataFrame(
            {"close": [50000.0, 51000.0]},
            index=pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
        )

        enriched = service_with_data.enrich(
            ohlcv, "deribit", "btc_pc_ratio", columns=["pc_ratio"], lag_days=0
        )
        assert "pc_ratio" in enriched.columns
        assert len(enriched) == 2

    def test_enrich_backward_merge(self, service_with_data: OptionsDataService) -> None:
        """merge_asof backward 방향 검증 (pc_ratio)."""
        ohlcv = pd.DataFrame(
            {"price": [50000.0, 51000.0, 52000.0]},
            index=pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"], utc=True),
        )

        enriched = service_with_data.enrich(
            ohlcv, "deribit", "btc_pc_ratio", columns=["pc_ratio"], lag_days=0
        )
        # 1/17에는 1/16 데이터가 forward-fill (backward merge)
        assert not pd.isna(enriched.iloc[2]["pc_ratio"])

    def test_batch_definitions_constant(self) -> None:
        """OPTIONS_BATCH_DEFINITIONS 상수 확인."""
        assert "deribit" in OPTIONS_BATCH_DEFINITIONS
        assert len(OPTIONS_BATCH_DEFINITIONS["deribit"]) == 6
