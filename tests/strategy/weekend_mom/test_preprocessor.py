"""Tests for Weekend-Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.weekend_mom.config import WeekendMomConfig
from src.strategy.weekend_mom.preprocessor import preprocess


@pytest.fixture
def config() -> WeekendMomConfig:
    return WeekendMomConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "weighted_returns",
            "is_weekend",
            "fast_mom",
            "slow_mom",
            "mom_score",
            "drawdown",
            "atr",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: WeekendMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_weekend_boost_applied(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """Weekend bars get boosted returns."""
        config_boost = WeekendMomConfig(weekend_boost=3.0)
        config_no_boost = WeekendMomConfig(weekend_boost=1.0)
        result_boost = preprocess(sample_ohlcv_df, config_boost)
        result_no_boost = preprocess(sample_ohlcv_df, config_no_boost)
        # Weekend bars should differ, weekday bars should be same
        is_weekend = result_boost["is_weekend"].astype(bool)
        weekend_mask = is_weekend & result_boost["returns"].notna()
        weekday_mask = ~is_weekend & result_boost["returns"].notna()
        if weekend_mask.any():
            np.testing.assert_array_almost_equal(
                result_boost["weighted_returns"][weekend_mask].values,
                result_no_boost["weighted_returns"][weekend_mask].values * 3.0,
            )
        if weekday_mask.any():
            np.testing.assert_array_almost_equal(
                result_boost["weighted_returns"][weekday_mask].values,
                result_no_boost["weighted_returns"][weekday_mask].values,
            )

    def test_is_weekend_binary(
        self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert set(result["is_weekend"].unique()).issubset({0.0, 1.0})

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_atr_positive(self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["atr"].dropna()
        assert (valid > 0).all()

    def test_fast_slow_mom_computed(
        self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        # After warmup, should have values
        fast_valid = result["fast_mom"].dropna()
        slow_valid = result["slow_mom"].dropna()
        assert len(fast_valid) > 0
        assert len(slow_valid) > 0


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
