"""Tests for FR-Pred preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.fr_pred.config import FRPredConfig
from src.strategy.fr_pred.preprocessor import preprocess


@pytest.fixture
def config() -> FRPredConfig:
    return FRPredConfig()


@pytest.fixture
def sample_deriv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.randn(n) * 0.0005  # typical FR range
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
    def test_output_columns(self, sample_deriv_df: pd.DataFrame, config: FRPredConfig) -> None:
        result = preprocess(sample_deriv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "fr_ma",
            "fr_zscore",
            "fr_mom_cross",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_deriv_df: pd.DataFrame, config: FRPredConfig) -> None:
        result = preprocess(sample_deriv_df, config)
        assert len(result) == len(sample_deriv_df)

    def test_immutability(self, sample_deriv_df: pd.DataFrame, config: FRPredConfig) -> None:
        original = sample_deriv_df.copy()
        preprocess(sample_deriv_df, config)
        pd.testing.assert_frame_equal(sample_deriv_df, original)

    def test_missing_columns(self, config: FRPredConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: FRPredConfig) -> None:
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "open": np.random.randn(n),
                "high": np.random.randn(n),
                "low": np.random.randn(n),
                "close": np.random.randn(n),
                "volume": np.random.randn(n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1D"),
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_deriv_df: pd.DataFrame, config: FRPredConfig) -> None:
        result = preprocess(sample_deriv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_funding_rate_ffill(self, config: FRPredConfig) -> None:
        """NaN이 포함된 funding_rate가 ffill로 처리되는지 확인."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)
        open_ = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 10000, n).astype(float)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        fr = np.random.randn(n) * 0.0005
        fr[:10] = np.nan  # first 10 are NaN
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "funding_rate": fr,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1D"),
        )
        result = preprocess(df, config)
        assert len(result) == n
