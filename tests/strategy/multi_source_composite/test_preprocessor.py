"""Tests for Multi-Source Directional Composite preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.multi_source_composite.config import MultiSourceCompositeConfig
from src.strategy.multi_source_composite.preprocessor import preprocess


@pytest.fixture
def config() -> MultiSourceCompositeConfig:
    return MultiSourceCompositeConfig()


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
    # F&G: 0~100 range with some NaN
    fg = np.clip(50 + np.cumsum(np.random.randn(n) * 3), 0, 100).astype(float)
    fg[:5] = np.nan  # Initial NaN to test ffill
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_fear_greed": fg,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "ema_fast",
            "ema_slow",
            "price_mom",
            "mom_direction",
            "velocity_fast",
            "velocity_slow",
            "velocity_direction",
            "fg_delta_smooth",
            "fg_direction",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: MultiSourceCompositeConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_fear_greed_column(self, config: MultiSourceCompositeConfig) -> None:
        df = pd.DataFrame(
            {
                "open": [1, 2],
                "high": [2, 3],
                "low": [0.5, 1.5],
                "close": [1.5, 2.5],
                "volume": [100, 200],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_mom_direction_values(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["mom_direction"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_velocity_direction_values(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["velocity_direction"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_fg_direction_values(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["fg_direction"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_fg_ffill_handles_nan(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        """oc_fear_greed NaN이 ffill로 처리되어 fg_delta_smooth 계산에 영향 없음."""
        result = preprocess(sample_ohlcv_df, config)
        # After warmup, fg_delta_smooth should have valid values
        warmup = config.fg_delta_window + config.fg_smooth_window + 10
        valid_region = result["fg_delta_smooth"].iloc[warmup:]
        assert valid_region.notna().any()

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_returns_finite(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["returns"].dropna()
        assert np.isfinite(valid).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
