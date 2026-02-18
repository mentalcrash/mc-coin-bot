"""Tests for Funding Pressure Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.fr_press_trend.config import FrPressTrendConfig
from src.strategy.fr_press_trend.preprocessor import preprocess


@pytest.fixture
def config() -> FrPressTrendConfig:
    return FrPressTrendConfig()


@pytest.fixture
def sample_ohlcv_fr_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.randn(n) * 0.0003
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_fr_df: pd.DataFrame, config: FrPressTrendConfig) -> None:
        result = preprocess(sample_ohlcv_fr_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "sma_fast",
            "sma_slow",
            "er",
            "avg_fr",
            "fr_z",
            "drawdown",
            "atr",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_fr_df: pd.DataFrame, config: FrPressTrendConfig) -> None:
        result = preprocess(sample_ohlcv_fr_df, config)
        assert len(result) == len(sample_ohlcv_fr_df)

    def test_immutability(self, sample_ohlcv_fr_df: pd.DataFrame, config: FrPressTrendConfig) -> None:
        original = sample_ohlcv_fr_df.copy()
        preprocess(sample_ohlcv_fr_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_fr_df, original)

    def test_missing_columns(self, config: FrPressTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: FrPressTrendConfig) -> None:
        df = pd.DataFrame({
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1, 2],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 200, 300],
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_fr_df: pd.DataFrame, config: FrPressTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_fr_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_er_range(self, sample_ohlcv_fr_df: pd.DataFrame, config: FrPressTrendConfig) -> None:
        result = preprocess(sample_ohlcv_fr_df, config)
        valid = result["er"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.0).all()
