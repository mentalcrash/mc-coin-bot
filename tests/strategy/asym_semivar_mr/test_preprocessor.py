"""Tests for Asymmetric Semivariance MR preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.asym_semivar_mr.config import AsymSemivarMRConfig
from src.strategy.asym_semivar_mr.preprocessor import preprocess


@pytest.fixture
def config() -> AsymSemivarMRConfig:
    return AsymSemivarMRConfig()


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
        self, sample_ohlcv_df: pd.DataFrame, config: AsymSemivarMRConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "down_semivar",
            "up_semivar",
            "semivar_ratio",
            "semivar_zscore",
            "atr",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: AsymSemivarMRConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: AsymSemivarMRConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: AsymSemivarMRConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_returns_log(self, sample_ohlcv_df: pd.DataFrame, config: AsymSemivarMRConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        # Log return check: returns = log(close[t] / close[t-1])
        expected = np.log(sample_ohlcv_df["close"].iloc[1] / sample_ohlcv_df["close"].iloc[0])
        assert abs(result["returns"].iloc[1] - expected) < 1e-10

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: AsymSemivarMRConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()


class TestSemivariance:
    def test_semivar_non_negative(
        self, sample_ohlcv_df: pd.DataFrame, config: AsymSemivarMRConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_down = result["down_semivar"].dropna()
        valid_up = result["up_semivar"].dropna()
        assert (valid_down >= 0).all()
        assert (valid_up >= 0).all()

    def test_semivar_ratio_range(
        self, sample_ohlcv_df: pd.DataFrame, config: AsymSemivarMRConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["semivar_ratio"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_semivar_ratio_balanced_market(self, config: AsymSemivarMRConfig) -> None:
        """Symmetric returns should give ratio near 0.5."""
        np.random.seed(123)
        n = 300
        # Symmetric returns around 0
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)  # Ensure positive prices
        high = close + 1.0
        low = close - 1.0
        open_ = close + 0.1
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        result = preprocess(df, config)
        valid_ratio = result["semivar_ratio"].dropna()
        if len(valid_ratio) > 0:
            # Symmetric returns -> ratio should be roughly 0.5
            mean_ratio = valid_ratio.mean()
            assert 0.3 < mean_ratio < 0.7

    def test_drawdown_always_nonpositive(
        self, sample_ohlcv_df: pd.DataFrame, config: AsymSemivarMRConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_zscore_has_values(
        self, sample_ohlcv_df: pd.DataFrame, config: AsymSemivarMRConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["semivar_zscore"].dropna()
        assert len(valid) > 0

    def test_doji_bars_handled(self, config: AsymSemivarMRConfig) -> None:
        """All same close -> semivar ratio is NaN (total = 0)."""
        n = 200
        close = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        open_ = np.full(n, 100.0)
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        result = preprocess(df, config)
        # Returns are all ~0 -> semivariances ~0 -> ratio NaN
        # This should not raise any error
        assert len(result) == n
