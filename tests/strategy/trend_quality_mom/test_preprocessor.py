"""Tests for Trend Quality Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.trend_quality_mom.config import TrendQualityMomConfig
from src.strategy.trend_quality_mom.preprocessor import preprocess


@pytest.fixture
def config() -> TrendQualityMomConfig:
    return TrendQualityMomConfig()


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
        self, sample_ohlcv_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "reg_slope",
            "r_squared",
            "mom_return",
            "atr",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: TrendQualityMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()


class TestRSquared:
    def test_r_squared_range(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["r_squared"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_perfect_trend(self) -> None:
        """Perfect linear trend should have R^2 close to 1."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="4h")
        close = np.linspace(100, 200, n)
        high = close + 1
        low = close - 1
        open_ = close - 0.5
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
        config = TrendQualityMomConfig(regression_lookback=20)
        result = preprocess(df, config)
        valid_r2 = result["r_squared"].dropna()
        assert len(valid_r2) > 0
        assert valid_r2.iloc[-1] > 0.95

    def test_slope_sign(self, sample_ohlcv_df: pd.DataFrame, config: TrendQualityMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        # Slope should be numeric (float)
        valid_slope = result["reg_slope"].dropna()
        assert len(valid_slope) > 0


class TestDrawdown:
    def test_drawdown_range(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()
