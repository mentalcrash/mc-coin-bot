"""Tests for KAMA Efficiency Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.kama_eff_trend.config import KamaEffTrendConfig
from src.strategy.kama_eff_trend.preprocessor import preprocess


@pytest.fixture
def config() -> KamaEffTrendConfig:
    return KamaEffTrendConfig()


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
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "kama_line",
            "er",
            "kama_slope",
            "kama_dist",
            "atr",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: KamaEffTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_er_range(self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_er = result["er"].dropna()
        assert (valid_er >= 0).all()
        assert (valid_er <= 1).all()

    def test_kama_line_not_all_nan(
        self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result["kama_line"].notna().sum() > 0

    def test_kama_slope_exists(
        self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["kama_slope"].dropna()
        assert len(valid) > 0

    def test_drawdown_range(
        self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
