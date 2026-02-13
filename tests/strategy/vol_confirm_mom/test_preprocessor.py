"""Tests for Volume-Confirmed Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_confirm_mom.config import VolConfirmMomConfig
from src.strategy.vol_confirm_mom.preprocessor import preprocess


@pytest.fixture
def config() -> VolConfirmMomConfig:
    return VolConfirmMomConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "mom_return",
            "vol_sma_short",
            "vol_sma_long",
            "vol_rising",
            "vol_ratio",
            "atr",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: VolConfirmMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()


class TestVolumeFeatures:
    def test_vol_rising_bool(
        self, sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_rising"].dropna()
        assert valid.dtype == bool

    def test_vol_ratio_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_ratio"].dropna()
        assert (valid > 0).all()

    def test_vol_ratio_clipped(
        self, sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_ratio"].dropna()
        assert (valid <= config.vol_ratio_clip).all()

    def test_vol_sma_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_short = result["vol_sma_short"].dropna()
        valid_long = result["vol_sma_long"].dropna()
        assert (valid_short > 0).all()
        assert (valid_long > 0).all()


class TestDrawdown:
    def test_drawdown_range(
        self, sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()
