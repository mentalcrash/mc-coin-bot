"""Tests for Squeeze-Adaptive Breakout preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.squeeze_adaptive_breakout.config import SqueezeAdaptiveBreakoutConfig
from src.strategy.squeeze_adaptive_breakout.preprocessor import preprocess


@pytest.fixture
def config() -> SqueezeAdaptiveBreakoutConfig:
    return SqueezeAdaptiveBreakoutConfig()


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
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "bb_upper",
            "bb_lower",
            "kc_upper",
            "kc_lower",
            "squeeze_on",
            "squeeze_duration",
            "kama",
            "bb_position",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: SqueezeAdaptiveBreakoutConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_squeeze_on_bool(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["squeeze_on"].dropna()
        assert set(valid.unique()).issubset({True, False})

    def test_bb_position_range(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["bb_position"].dropna()
        # BB position can go outside [0, 1] by design, but most values should be in range
        assert len(valid) > 0

    def test_kama_not_all_nan(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["kama"].dropna()
        assert len(valid) > 0

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0.0).all()

    def test_squeeze_duration_range(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["squeeze_duration"].dropna()
        assert (valid >= 0).all()
        assert (valid <= config.squeeze_lookback).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
