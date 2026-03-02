"""Tests for LR-Channel Multi-Scale Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.lr_channel_trend.config import LrChannelTrendConfig
from src.strategy.lr_channel_trend.preprocessor import preprocess


@pytest.fixture
def config() -> LrChannelTrendConfig:
    return LrChannelTrendConfig()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


class TestPreprocessColumns:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        """전처리 후 lr_upper/lr_lower + vol 컬럼이 추가되는지 확인."""
        result = preprocess(sample_ohlcv_df, config)
        scales = (config.scale_short, config.scale_mid, config.scale_long)
        for s in scales:
            assert f"lr_upper_{s}" in result.columns, f"lr_upper_{s} missing"
            assert f"lr_lower_{s}" in result.columns, f"lr_lower_{s} missing"
        assert "returns" in result.columns
        assert "realized_vol" in result.columns
        assert "vol_scalar" in result.columns
        assert "drawdown" in result.columns

    def test_same_length(
        self, sample_ohlcv_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: LrChannelTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_lte_zero(
        self, sample_ohlcv_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_lr_upper_gte_lower(
        self, sample_ohlcv_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        """LR upper >= lower for all scales."""
        result = preprocess(sample_ohlcv_df, config)
        scales = (config.scale_short, config.scale_mid, config.scale_long)
        for s in scales:
            upper = result[f"lr_upper_{s}"].dropna()
            lower = result[f"lr_lower_{s}"].dropna()
            assert (upper >= lower).all(), f"lr_upper_{s} < lr_lower_{s} found"

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_preserves_original_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_custom_scales(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """커스텀 스케일로 전처리 시 올바른 컬럼 추가."""
        config = LrChannelTrendConfig(scale_short=10, scale_mid=30, scale_long=80)
        result = preprocess(sample_ohlcv_df, config)
        assert "lr_upper_10" in result.columns
        assert "lr_lower_30" in result.columns
        assert "lr_upper_80" in result.columns
        # 기본 스케일 컬럼은 없어야 함
        assert "lr_upper_20" not in result.columns
        assert "lr_upper_150" not in result.columns
