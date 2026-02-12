"""Tests for Capitulation Wick Reversal preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.cap_wick_rev.config import CapWickRevConfig
from src.strategy.cap_wick_rev.preprocessor import preprocess


@pytest.fixture
def config() -> CapWickRevConfig:
    return CapWickRevConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "current_atr",
            "atr_ratio",
            "vol_ratio",
            "lower_wick_ratio",
            "upper_wick_ratio",
            "close_position",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: CapWickRevConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_returns_log(self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        expected = np.log(sample_ohlcv_df["close"].iloc[1] / sample_ohlcv_df["close"].iloc[0])
        assert abs(result["returns"].iloc[1] - expected) < 1e-10

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()


class TestWickRatios:
    def test_wick_ratios_range(
        self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        lower = result["lower_wick_ratio"].dropna()
        upper = result["upper_wick_ratio"].dropna()
        assert (lower >= 0).all()
        assert (lower <= 1).all()
        assert (upper >= 0).all()
        assert (upper <= 1).all()

    def test_close_position_range(
        self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["close_position"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_atr_ratio_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["atr_ratio"].dropna()
        assert (valid > 0).all()

    def test_vol_ratio_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_ratio"].dropna()
        assert (valid > 0).all()


class TestEdgeCases:
    def test_doji_bars(self, config: CapWickRevConfig) -> None:
        """All doji bars (open == close == high == low) -> NaN wick ratios."""
        n = 200
        price = np.full(n, 100.0)
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        result = preprocess(df, config)
        # Doji -> range=0 -> wick ratios are NaN
        assert len(result) == n

    def test_drawdown_always_nonpositive(
        self, sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_large_lower_wick_bar(self, config: CapWickRevConfig) -> None:
        """Bar with large lower wick should have high lower_wick_ratio."""
        n = 100
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 1.0
        low = close - 1.0
        open_ = close + 0.1
        volume = np.full(n, 5000.0)

        # Create one bar with a very large lower wick at index 50
        low[50] = close[50] - 20.0
        high[50] = close[50] + 1.0

        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        result = preprocess(df, config)
        # Bar 50 should have a high lower_wick_ratio
        lw = result["lower_wick_ratio"].iloc[50]
        assert not np.isnan(lw)
        assert lw > 0.5
