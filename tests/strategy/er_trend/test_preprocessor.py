"""Tests for ER Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.er_trend.config import ErTrendConfig
from src.strategy.er_trend.preprocessor import preprocess


@pytest.fixture
def config() -> ErTrendConfig:
    return ErTrendConfig()


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


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "signed_er_fast",
            "signed_er_mid",
            "signed_er_slow",
            "composite_ser",
            "atr",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: ErTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_signed_er_range(self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig) -> None:
        """Signed ER은 [-1, +1] 범위여야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        for col in ["signed_er_fast", "signed_er_mid", "signed_er_slow"]:
            valid = result[col].dropna()
            assert (valid >= -1.0 - 1e-10).all(), f"{col} has values below -1"
            assert (valid <= 1.0 + 1e-10).all(), f"{col} has values above 1"

    def test_composite_ser_is_weighted_sum(
        self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig
    ) -> None:
        """composite_ser = w_fast*fast + w_mid*mid + w_slow*slow."""
        result = preprocess(sample_ohlcv_df, config)
        expected = (
            config.w_fast * result["signed_er_fast"]
            + config.w_mid * result["signed_er_mid"]
            + config.w_slow * result["signed_er_slow"]
        )
        pd.testing.assert_series_equal(result["composite_ser"], expected, check_names=False)

    def test_composite_ser_range(
        self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig
    ) -> None:
        """composite_ser은 [-1, +1] 범위여야 한다 (weights sum=1이므로)."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["composite_ser"].dropna()
        assert (valid >= -1.0 - 1e-10).all()
        assert (valid <= 1.0 + 1e-10).all()

    def test_drawdown_negative_or_zero(
        self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0.0 + 1e-10).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
