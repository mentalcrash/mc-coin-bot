"""Tests for Z-Momentum (MACD-V) preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.z_mom.config import ZMomConfig
from src.strategy.z_mom.preprocessor import preprocess


@pytest.fixture
def config() -> ZMomConfig:
    return ZMomConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "macd_v",
            "macd_v_signal",
            "macd_v_hist",
            "mom_return",
            "atr",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: ZMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_macd_v_not_all_nan(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert not result["macd_v"].isna().all()

    def test_macd_v_hist_not_all_nan(
        self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert not result["macd_v_hist"].isna().all()

    def test_atr_positive(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_atr = result["atr"].dropna()
        assert (valid_atr > 0).all()

    def test_drawdown_non_positive(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0).all()

    def test_custom_config(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ZMomConfig(macd_fast=8, macd_slow=21, macd_signal=5)
        result = preprocess(sample_ohlcv_df, config)
        assert "macd_v" in result.columns
        assert "macd_v_hist" in result.columns

    def test_mom_return_column(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert "mom_return" in result.columns
        assert not result["mom_return"].isna().all()


class TestPreprocessorImmutability:
    def test_original_unchanged(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
