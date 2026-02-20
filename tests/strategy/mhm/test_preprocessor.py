"""Tests for MHM preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mhm.config import MHMConfig
from src.strategy.mhm.preprocessor import preprocess


@pytest.fixture
def config() -> MHMConfig:
    return MHMConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: MHMConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "weighted_mom",
            "max_agreement",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_momentum_columns(self, sample_ohlcv_df: pd.DataFrame, config: MHMConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        for i in range(1, 6):
            assert f"mom_{i}" in result.columns
            assert f"mom_vol_{i}" in result.columns

    def test_agreement_columns(self, sample_ohlcv_df: pd.DataFrame, config: MHMConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert "pos_agreement" in result.columns
        assert "neg_agreement" in result.columns
        # max_agreement should be between 0 and 5
        valid = result["max_agreement"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 5).all()

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: MHMConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: MHMConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: MHMConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame, config: MHMConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
