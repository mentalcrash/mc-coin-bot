"""Tests for FgPersistBreak preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market.indicators import count_consecutive
from src.strategy.fg_persist_break.config import FgPersistBreakConfig
from src.strategy.fg_persist_break.preprocessor import preprocess


@pytest.fixture
def config() -> FgPersistBreakConfig:
    return FgPersistBreakConfig()


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
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )
    df["oc_fear_greed"] = np.random.randint(10, 90, n).astype(float)
    return df


class TestCountConsecutive:
    def test_basic(self) -> None:
        mask = np.array([False, True, True, True, False, True, True])
        result = count_consecutive(mask)
        np.testing.assert_array_equal(result, [0, 1, 2, 3, 0, 1, 2])

    def test_all_false(self) -> None:
        mask = np.array([False, False, False])
        result = count_consecutive(mask)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_all_true(self) -> None:
        mask = np.array([True, True, True])
        result = count_consecutive(mask)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_empty(self) -> None:
        result = count_consecutive(np.array([], dtype=bool))
        assert len(result) == 0


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: FgPersistBreakConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "fear_streak",
            "greed_streak",
            "price_mom",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: FgPersistBreakConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: FgPersistBreakConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: FgPersistBreakConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: FgPersistBreakConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_streak_nonnegative(
        self, sample_ohlcv_df: pd.DataFrame, config: FgPersistBreakConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert (result["fear_streak"] >= 0).all()
        assert (result["greed_streak"] >= 0).all()

    def test_fear_streak_in_fear_zone(self, config: FgPersistBreakConfig) -> None:
        """Fear zone 내에서만 streak > 0."""
        n = 20
        df = pd.DataFrame(
            {
                "open": np.ones(n) * 100,
                "high": np.ones(n) * 101,
                "low": np.ones(n) * 99,
                "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
                "oc_fear_greed": [10.0] * 8 + [50.0] * 12,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1D"),
        )
        result = preprocess(df, config)
        # First 8 bars: fear zone (F&G=10 < 25) → streak 1,2,...,8
        assert result["fear_streak"].iloc[7] == 8
        # After exit: streak resets
        assert result["fear_streak"].iloc[8] == 0
