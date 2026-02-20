"""Tests for Volatility Surface Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_surface_mom.config import VolSurfaceMomConfig
from src.strategy.vol_surface_mom.preprocessor import preprocess


@pytest.fixture
def config() -> VolSurfaceMomConfig:
    return VolSurfaceMomConfig()


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
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: VolSurfaceMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "gk_vol",
            "pk_vol",
            "yz_vol",
            "gk_pk_ratio",
            "yz_pk_ratio",
            "momentum",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: VolSurfaceMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: VolSurfaceMomConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: VolSurfaceMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VolSurfaceMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_gk_pk_ratio_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VolSurfaceMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["gk_pk_ratio"].dropna()
        assert len(valid) > 0

    def test_drawdown_leq_zero(
        self, sample_ohlcv_df: pd.DataFrame, config: VolSurfaceMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_pk_vol_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VolSurfaceMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["pk_vol"].dropna()
        assert (valid > 0).all()
