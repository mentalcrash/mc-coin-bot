"""Tests for T-Stat Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.t_stat_mom.config import TStatMomConfig
from src.strategy.t_stat_mom.preprocessor import preprocess


@pytest.fixture
def config() -> TStatMomConfig:
    return TStatMomConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "t_stat_fast",
            "t_stat_mid",
            "t_stat_slow",
            "t_stat_blend",
            "drawdown",
            "atr",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: TStatMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_t_stat_blend_is_average(
        self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        expected = (result["t_stat_fast"] + result["t_stat_mid"] + result["t_stat_slow"]) / 3.0
        pd.testing.assert_series_equal(result["t_stat_blend"], expected, check_names=False)

    def test_t_stat_warmup_nan(self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig) -> None:
        """Warmup 기간 내 t-stat은 NaN이어야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        # slow_lookback까지는 NaN
        assert result["t_stat_slow"].iloc[: config.slow_lookback - 1].isna().all()

    def test_t_stat_after_warmup_finite(
        self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig
    ) -> None:
        """Warmup 이후 t-stat은 유한값이어야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        after_warmup = result["t_stat_blend"].iloc[config.slow_lookback :]
        assert after_warmup.notna().all()
        assert np.isfinite(after_warmup).all()

    def test_custom_lookbacks(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TStatMomConfig(fast_lookback=10, mid_lookback=30, slow_lookback=60)
        result = preprocess(sample_ohlcv_df, config)
        assert result["t_stat_fast"].iloc[10:].notna().all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
