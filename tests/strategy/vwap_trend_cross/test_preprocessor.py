"""Tests for VWAP Trend Crossover preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vwap_trend_cross.config import VwapTrendCrossConfig
from src.strategy.vwap_trend_cross.preprocessor import preprocess


@pytest.fixture
def config() -> VwapTrendCrossConfig:
    return VwapTrendCrossConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "vwap_short",
            "vwap_long",
            "vwap_spread",
            "atr",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: VwapTrendCrossConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: VwapTrendCrossConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()


class TestVwapFeatures:
    def test_vwap_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_short = result["vwap_short"].dropna()
        valid_long = result["vwap_long"].dropna()
        assert (valid_short > 0).all()
        assert (valid_long > 0).all()

    def test_vwap_spread_clipped(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vwap_spread"].dropna()
        assert (valid >= -config.spread_clip).all()
        assert (valid <= config.spread_clip).all()

    def test_vwap_near_close(self) -> None:
        """VWAP should be near the close price level."""
        n = 100
        close = pd.Series(np.full(n, 100.0))
        high = close + 1
        low = close - 1
        open_ = close
        volume = pd.Series(np.full(n, 5000.0))
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        config = VwapTrendCrossConfig(vwap_short_window=10, vwap_long_window=30)
        result = preprocess(df, config)
        valid_short = result["vwap_short"].dropna()
        # With constant price and volume, VWAP should equal close
        np.testing.assert_allclose(valid_short.values, 100.0, atol=0.01)


class TestDrawdown:
    def test_drawdown_range(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()
