"""Tests for DerivExtDataService."""

from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.data.deriv_ext.service import DERIV_EXT_BATCH_DEFINITIONS, DerivExtDataService
from src.data.deriv_ext.storage import DerivExtBronzeStorage, DerivExtSilverProcessor


@pytest.fixture
def settings(tmp_path: Path) -> IngestionSettings:
    """테스트용 설정."""
    return IngestionSettings(
        deriv_ext_bronze_dir=tmp_path / "bronze" / "deriv_ext",
        deriv_ext_silver_dir=tmp_path / "silver" / "deriv_ext",
    )


@pytest.fixture
def service_with_data(settings: IngestionSettings) -> DerivExtDataService:
    """Silver 데이터가 준비된 서비스."""
    bronze = DerivExtBronzeStorage(settings)
    silver = DerivExtSilverProcessor(settings, bronze)

    # BTC Agg OI 데이터
    oi_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
            "symbol": ["BTCUSDT.6"] * 2,
            "close": [51000.0, 52000.0],
            "source": ["coinalyze"] * 2,
        }
    )
    bronze.save(oi_df, "coinalyze", "btc_agg_oi")
    silver.process("coinalyze", "btc_agg_oi")

    # BTC Agg Funding 데이터
    funding_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
            "symbol": ["BTCUSDT.6"] * 2,
            "close": [0.0001, 0.0002],
            "source": ["coinalyze"] * 2,
        }
    )
    bronze.save(funding_df, "coinalyze", "btc_agg_funding")
    silver.process("coinalyze", "btc_agg_funding")

    # BTC Liquidations 데이터 (daily 롤업 완료된 데이터)
    liq_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
            "symbol": ["BTCUSDT.6"] * 2,
            "long_volume": [180000.0, 200000.0],
            "short_volume": [140000.0, 150000.0],
            "source": ["coinalyze"] * 2,
        }
    )
    bronze.save(liq_df, "coinalyze", "btc_liquidations")
    silver.process("coinalyze", "btc_liquidations")

    # BTC CVD 데이터
    cvd_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
            "symbol": ["BTCUSDT.6"] * 2,
            "buy_volume": [28000.0, 30000.0],
            "source": ["coinalyze"] * 2,
        }
    )
    bronze.save(cvd_df, "coinalyze", "btc_cvd")
    silver.process("coinalyze", "btc_cvd")

    return DerivExtDataService(settings, silver, catalog=None)


class TestDerivExtDataService:
    """DerivExtDataService 테스트."""

    def test_batch_definitions_coinalyze(self, settings: IngestionSettings) -> None:
        """Coinalyze batch definitions."""
        service = DerivExtDataService(settings, catalog=None)
        defs = service.get_batch_definitions("coinalyze")
        assert len(defs) == 8
        assert all(s == "coinalyze" for s, _ in defs)

    def test_batch_definitions_all(self, settings: IngestionSettings) -> None:
        """전체 batch definitions."""
        service = DerivExtDataService(settings, catalog=None)
        defs = service.get_batch_definitions("all")
        assert len(defs) == 10

    def test_batch_definitions_invalid(self, settings: IngestionSettings) -> None:
        """잘못된 batch type."""
        service = DerivExtDataService(settings, catalog=None)
        with pytest.raises(ValueError, match="Unknown batch type"):
            service.get_batch_definitions("invalid")

    def test_load(self, service_with_data: DerivExtDataService) -> None:
        """Silver 로드."""
        df = service_with_data.load("coinalyze", "btc_agg_oi")
        assert len(df) == 2

    def test_enrich(self, service_with_data: DerivExtDataService) -> None:
        """OHLCV에 deriv_ext 데이터 병합."""
        ohlcv = pd.DataFrame(
            {"close": [50000.0, 51000.0]},
            index=pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True),
        )

        enriched = service_with_data.enrich(
            ohlcv,
            "coinalyze",
            "btc_liquidations",
            columns=["long_volume", "short_volume"],
            lag_days=0,
        )
        assert "long_volume" in enriched.columns
        assert "short_volume" in enriched.columns
        assert len(enriched) == 2

    def test_enrich_backward_merge(self, service_with_data: DerivExtDataService) -> None:
        """merge_asof backward 방향 검증."""
        ohlcv = pd.DataFrame(
            {"price": [50000.0, 51000.0, 52000.0]},
            index=pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"], utc=True),
        )

        enriched = service_with_data.enrich(
            ohlcv,
            "coinalyze",
            "btc_agg_oi",
            columns=["close"],
            lag_days=0,
        )
        # 1/17에는 1/16 데이터가 forward-fill (backward merge)
        assert not pd.isna(enriched.iloc[2]["close"])

    def test_precompute_btc(self, service_with_data: DerivExtDataService) -> None:
        """BTC precompute — dext_* 컬럼 생성."""
        ohlcv_index = pd.to_datetime(["2024-01-15", "2024-01-16"], utc=True)
        result = service_with_data.precompute(ohlcv_index, asset="BTC")

        assert "dext_agg_oi_close" in result.columns
        assert "dext_agg_funding_close" in result.columns
        assert "dext_liq_long_vol" in result.columns
        assert "dext_liq_short_vol" in result.columns
        assert "dext_cvd_buy_vol" in result.columns
        assert len(result) == 2

    def test_batch_definitions_constant(self) -> None:
        """DERIV_EXT_BATCH_DEFINITIONS 상수 확인."""
        assert "coinalyze" in DERIV_EXT_BATCH_DEFINITIONS
        assert len(DERIV_EXT_BATCH_DEFINITIONS["coinalyze"]) == 8
