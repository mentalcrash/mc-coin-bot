"""Tests for FgEmaCycle preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.fg_ema_cycle.config import FgEmaCycleConfig
from src.strategy.fg_ema_cycle.preprocessor import preprocess


@pytest.fixture
def config() -> FgEmaCycleConfig:
    return FgEmaCycleConfig()


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
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )
    df["oc_fear_greed"] = np.random.randint(10, 90, n).astype(float)
    return df


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: FgEmaCycleConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "fg_ema_slow",
            "fg_ema_fast",
            "cycle_position",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: FgEmaCycleConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: FgEmaCycleConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: FgEmaCycleConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: FgEmaCycleConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_cycle_position_bounded(
        self, sample_ohlcv_df: pd.DataFrame, config: FgEmaCycleConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["cycle_position"].dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()
