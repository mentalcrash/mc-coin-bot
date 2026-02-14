"""Tests for Variance Decomposition Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vardecomp_mom.config import VardecompMomConfig
from src.strategy.vardecomp_mom.preprocessor import preprocess


@pytest.fixture
def config() -> VardecompMomConfig:
    return VardecompMomConfig()


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
        self, sample_ohlcv_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "good_semivar",
            "bad_semivar",
            "var_ratio",
            "price_mom",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: VardecompMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: VardecompMomConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: VardecompMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()


class TestSemivariance:
    def test_var_ratio_range(
        self, sample_ohlcv_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        """var_ratio는 0~1 범위."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["var_ratio"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_semivar_non_negative(
        self, sample_ohlcv_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        """Semivariance는 항상 0 이상."""
        result = preprocess(sample_ohlcv_df, config)
        good = result["good_semivar"].dropna()
        bad = result["bad_semivar"].dropna()
        assert (good >= 0).all()
        assert (bad >= 0).all()

    def test_price_mom_finite(
        self, sample_ohlcv_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["price_mom"].dropna()
        assert np.isfinite(valid).all()
