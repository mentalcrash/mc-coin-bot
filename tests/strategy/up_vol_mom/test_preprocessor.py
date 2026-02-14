"""Tests for up-vol-mom preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.up_vol_mom.config import UpVolMomConfig
from src.strategy.up_vol_mom.preprocessor import preprocess


@pytest.fixture
def config() -> UpVolMomConfig:
    return UpVolMomConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "up_semivar",
            "down_semivar",
            "up_ratio",
            "up_ratio_ma",
            "mom_direction",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: UpVolMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_present(self, sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert "drawdown" in result.columns
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_up_ratio_range(self, sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        """up_ratio is between 0 and 1 (exclusive NaN)."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["up_ratio"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_semivariance_nonnegative(
        self, sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig
    ) -> None:
        """Semivariance values should be non-negative (squared)."""
        result = preprocess(sample_ohlcv_df, config)
        up_valid = result["up_semivar"].dropna()
        down_valid = result["down_semivar"].dropna()
        assert (up_valid >= 0).all()
        assert (down_valid >= 0).all()

    def test_mom_direction_values(
        self, sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig
    ) -> None:
        """mom_direction should be -1, 0, or 1."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["mom_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_up_ratio_ma_smoother_than_raw(
        self, sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig
    ) -> None:
        """up_ratio_ma should be smoother (lower std) than up_ratio."""
        result = preprocess(sample_ohlcv_df, config)
        raw_std = result["up_ratio"].dropna().std()
        ma_std = result["up_ratio_ma"].dropna().std()
        assert ma_std <= raw_std
