"""Tests for Carry-Sentiment Gate preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.carry_sent.config import CarrySentConfig
from src.strategy.carry_sent.preprocessor import preprocess


@pytest.fixture
def config() -> CarrySentConfig:
    return CarrySentConfig()


@pytest.fixture
def sample_df() -> pd.DataFrame:
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
    fear_greed = np.random.randint(10, 90, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
            "oc_fear_greed": fear_greed,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_df: pd.DataFrame, config: CarrySentConfig) -> None:
        result = preprocess(sample_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "avg_funding_rate",
            "fr_zscore",
            "fg_ma",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_df: pd.DataFrame, config: CarrySentConfig) -> None:
        result = preprocess(sample_df, config)
        assert len(result) == len(sample_df)

    def test_immutability(self, sample_df: pd.DataFrame, config: CarrySentConfig) -> None:
        original = sample_df.copy()
        preprocess(sample_df, config)
        pd.testing.assert_frame_equal(sample_df, original)

    def test_missing_columns(self, config: CarrySentConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: CarrySentConfig) -> None:
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100.0],
                "oc_fear_greed": [50.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_fear_greed(self, config: CarrySentConfig) -> None:
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [100.0],
                "funding_rate": [0.001],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_df: pd.DataFrame, config: CarrySentConfig) -> None:
        result = preprocess(sample_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_fg_ma_range(self, sample_df: pd.DataFrame, config: CarrySentConfig) -> None:
        result = preprocess(sample_df, config)
        valid_fg_ma = result["fg_ma"].dropna()
        assert (valid_fg_ma >= 0).all()
        assert (valid_fg_ma <= 100).all()

    def test_drawdown_nonpositive(self, sample_df: pd.DataFrame, config: CarrySentConfig) -> None:
        result = preprocess(sample_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0).all()
