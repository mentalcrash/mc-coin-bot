"""Tests for TradeFlowService — Silver 로드 + enrichment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.settings import IngestionSettings
from src.data.trade_flow.service import TFLOW_COLUMNS, TradeFlowService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> IngestionSettings:
    """테스트용 IngestionSettings."""
    return IngestionSettings(
        trade_flow_silver_dir=tmp_path / "silver" / "trade_flow",
    )


@pytest.fixture
def service_with_data(settings: IngestionSettings) -> TradeFlowService:
    """Silver 데이터가 존재하는 서비스."""
    # 2024년 12H Silver 생성 (730 bars, 365 days x 2 bars/day)
    n_bars = 730
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="12h", tz="UTC")
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "tflow_cvd": rng.uniform(-0.3, 0.3, size=n_bars),
            "tflow_buy_ratio": rng.uniform(0.4, 0.6, size=n_bars),
            "tflow_intensity": rng.uniform(1000, 5000, size=n_bars),
            "tflow_large_ratio": rng.uniform(0.05, 0.15, size=n_bars),
            "tflow_abs_order_imbalance": rng.uniform(0.05, 0.3, size=n_bars),
            "tflow_vpin": rng.uniform(0.1, 0.4, size=n_bars),
        },
        index=dates,
    )
    df.index.name = "timestamp"

    path = settings.get_trade_flow_silver_path("BTC/USDT", 2024)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd")

    return TradeFlowService(settings)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTradeFlowService:
    def test_load_existing(self, service_with_data: TradeFlowService) -> None:
        """Silver 파일 로드 성공."""
        df = service_with_data.load("BTC/USDT", 2024)
        assert len(df) == 730
        assert "tflow_cvd" in df.columns
        assert "tflow_vpin" in df.columns

    def test_load_missing_raises(self, settings: IngestionSettings) -> None:
        """Silver 파일 없음 → FileNotFoundError."""
        service = TradeFlowService(settings)
        with pytest.raises(FileNotFoundError, match="Trade flow Silver not found"):
            service.load("ETH/USDT", 2024)

    def test_precompute_12h_index(self, service_with_data: TradeFlowService) -> None:
        """12H OHLCV 인덱스에 trade flow 피처 정렬."""
        ohlcv_index = pd.date_range("2024-03-01", "2024-06-30", freq="12h", tz="UTC")
        result = service_with_data.precompute(ohlcv_index, "BTC/USDT")

        assert len(result) == len(ohlcv_index)
        # tflow 컬럼 존재 확인
        for col in TFLOW_COLUMNS:
            assert col in result.columns

    def test_precompute_1d_index(self, service_with_data: TradeFlowService) -> None:
        """1D OHLCV 인덱스에도 merge_asof 동작."""
        ohlcv_index = pd.date_range("2024-03-01", "2024-06-30", freq="1D", tz="UTC")
        result = service_with_data.precompute(ohlcv_index, "BTC/USDT")

        assert len(result) == len(ohlcv_index)
        # backward merge → NaN은 데이터 이전 구간만
        assert result["tflow_cvd"].notna().sum() > 0

    def test_precompute_no_data(self, settings: IngestionSettings) -> None:
        """Silver 없음 → 빈 DataFrame (컬럼 없음)."""
        service = TradeFlowService(settings)
        ohlcv_index = pd.date_range("2024-01-01", periods=100, freq="12h", tz="UTC")
        result = service.precompute(ohlcv_index, "ETH/USDT")

        assert len(result) == len(ohlcv_index)
        assert result.columns.empty

    def test_precompute_empty_index(self, service_with_data: TradeFlowService) -> None:
        """빈 OHLCV 인덱스 → 빈 DataFrame."""
        result = service_with_data.precompute(pd.DatetimeIndex([]), "BTC/USDT")
        assert result.empty
