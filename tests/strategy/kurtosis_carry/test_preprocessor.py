"""Tests for Kurtosis Carry preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.kurtosis_carry.config import KurtosisCarryConfig
from src.strategy.kurtosis_carry.preprocessor import preprocess


@pytest.fixture
def config() -> KurtosisCarryConfig:
    return KurtosisCarryConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: KurtosisCarryConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "kurtosis_short",
            "kurtosis_long",
            "kurtosis_delta",
            "kurtosis_zscore",
            "momentum",
            "drawdown",
            "atr",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: KurtosisCarryConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: KurtosisCarryConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: KurtosisCarryConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: KurtosisCarryConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_kurtosis_finite(
        self, sample_ohlcv_df: pd.DataFrame, config: KurtosisCarryConfig
    ) -> None:
        """Non-NaN kurtosis values should be finite."""
        result = preprocess(sample_ohlcv_df, config)
        valid_short = result["kurtosis_short"].dropna()
        valid_long = result["kurtosis_long"].dropna()
        assert np.isfinite(valid_short).all()
        assert np.isfinite(valid_long).all()

    def test_momentum_values(
        self, sample_ohlcv_df: pd.DataFrame, config: KurtosisCarryConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["momentum"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})
