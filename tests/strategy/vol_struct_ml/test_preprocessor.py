"""Tests for Volatility Structure ML preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_struct_ml.config import VolStructMLConfig
from src.strategy.vol_struct_ml.preprocessor import preprocess


@pytest.fixture
def config() -> VolStructMLConfig:
    return VolStructMLConfig()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 400
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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: VolStructMLConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {"returns", "realized_vol", "vol_scalar", "forward_return"}
        assert required.issubset(set(result.columns))

    def test_feature_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: VolStructMLConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) == 13

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: VolStructMLConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: VolStructMLConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: VolStructMLConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VolStructMLConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_present(
        self, sample_ohlcv_df: pd.DataFrame, config: VolStructMLConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert "drawdown" in result.columns
