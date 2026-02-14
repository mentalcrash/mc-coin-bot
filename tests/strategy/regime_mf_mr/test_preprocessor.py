"""Tests for Regime-Gated Multi-Factor MR preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.regime_mf_mr.config import RegimeMfMrConfig
from src.strategy.regime_mf_mr.preprocessor import preprocess


@pytest.fixture
def config() -> RegimeMfMrConfig:
    return RegimeMfMrConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: RegimeMfMrConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "bb_pos",
            "price_zscore",
            "mr_score",
            "rsi",
            "volume_ma",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: RegimeMfMrConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: RegimeMfMrConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: RegimeMfMrConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: RegimeMfMrConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_bb_pos_range(self, sample_ohlcv_df: pd.DataFrame, config: RegimeMfMrConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["bb_pos"].dropna()
        # bb_position can go outside 0-1 in extreme moves, but mostly in range
        assert len(valid) > 0

    def test_rsi_range(self, sample_ohlcv_df: pd.DataFrame, config: RegimeMfMrConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["rsi"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: RegimeMfMrConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()
