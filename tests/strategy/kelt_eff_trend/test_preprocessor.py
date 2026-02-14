"""Tests for Keltner Efficiency Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.kelt_eff_trend.config import KeltEffTrendConfig
from src.strategy.kelt_eff_trend.preprocessor import preprocess


@pytest.fixture
def config() -> KeltEffTrendConfig:
    return KeltEffTrendConfig()


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
        self, sample_ohlcv_df: pd.DataFrame, config: KeltEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "kc_upper",
            "kc_middle",
            "kc_lower",
            "efficiency_ratio",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: KeltEffTrendConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: KeltEffTrendConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: KeltEffTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: KeltEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: KeltEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()


class TestKeltnerFeatures:
    def test_kc_upper_above_middle(
        self, sample_ohlcv_df: pd.DataFrame, config: KeltEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_upper = result["kc_upper"].dropna()
        valid_middle = result["kc_middle"].dropna()
        common = valid_upper.index.intersection(valid_middle.index)
        assert (valid_upper.loc[common] >= valid_middle.loc[common]).all()

    def test_kc_lower_below_middle(
        self, sample_ohlcv_df: pd.DataFrame, config: KeltEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_lower = result["kc_lower"].dropna()
        valid_middle = result["kc_middle"].dropna()
        common = valid_lower.index.intersection(valid_middle.index)
        assert (valid_lower.loc[common] <= valid_middle.loc[common]).all()

    def test_efficiency_ratio_range(
        self, sample_ohlcv_df: pd.DataFrame, config: KeltEffTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["efficiency_ratio"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.0 + 1e-6).all()
