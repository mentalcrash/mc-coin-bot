"""Tests for FR Quality Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.fr_quality_mom.config import FrQualityMomConfig
from src.strategy.fr_quality_mom.preprocessor import preprocess


@pytest.fixture
def config() -> FrQualityMomConfig:
    return FrQualityMomConfig()


@pytest.fixture
def sample_ohlcv_with_funding_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.uniform(-0.001, 0.001, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FrQualityMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "momentum",
            "avg_funding_rate",
            "fr_zscore",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FrQualityMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        assert len(result) == len(sample_ohlcv_with_funding_df)

    def test_immutability(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FrQualityMomConfig
    ) -> None:
        original = sample_ohlcv_with_funding_df.copy()
        preprocess(sample_ohlcv_with_funding_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_with_funding_df, original)

    def test_missing_columns(self, config: FrQualityMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: FrQualityMomConfig) -> None:
        df = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [2, 3, 4],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
                "volume": [100, 200, 300],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FrQualityMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
