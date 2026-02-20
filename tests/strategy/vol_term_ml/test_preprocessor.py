"""Tests for Vol-Term ML preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_term_ml.config import VolTermMLConfig
from src.strategy.vol_term_ml.preprocessor import preprocess


@pytest.fixture
def config() -> VolTermMLConfig:
    return VolTermMLConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: VolTermMLConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {"returns", "realized_vol", "vol_scalar", "forward_return"}
        assert required.issubset(set(result.columns))
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) == 10

    def test_rv_windows(self, sample_ohlcv_df: pd.DataFrame, config: VolTermMLConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        for w in [5, 10, 20, 40, 60]:
            assert f"feat_rv_{w}" in result.columns

    def test_vol_ratios(self, sample_ohlcv_df: pd.DataFrame, config: VolTermMLConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert "feat_vol_ratio_5_20" in result.columns
        assert "feat_vol_ratio_10_40" in result.columns
        assert "feat_vol_ratio_20_60" in result.columns

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: VolTermMLConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: VolTermMLConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: VolTermMLConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_empty_dataframe(self, config: VolTermMLConfig) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="empty"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VolTermMLConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
