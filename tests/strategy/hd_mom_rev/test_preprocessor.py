"""Tests for hd-mom-rev preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.hd_mom_rev.config import HdMomRevConfig
from src.strategy.hd_mom_rev.preprocessor import preprocess


@pytest.fixture
def config() -> HdMomRevConfig:
    return HdMomRevConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "half_return",
            "half_return_smooth",
            "jump_score",
            "is_jump",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: HdMomRevConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_present(self, sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert "drawdown" in result.columns
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_half_return_computable(
        self, sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig
    ) -> None:
        """half_return = log(close / open) should be finite."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["half_return"].dropna()
        assert np.isfinite(valid).all()

    def test_jump_score_nonnegative(
        self, sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig
    ) -> None:
        """jump_score should be non-negative (absolute value based)."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["jump_score"].dropna()
        assert (valid >= 0).all()

    def test_is_jump_boolean(self, sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        """is_jump should be boolean."""
        result = preprocess(sample_ohlcv_df, config)
        assert result["is_jump"].dtype == bool

    def test_half_return_smooth_smoother(
        self, sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig
    ) -> None:
        """half_return_smooth should be smoother than raw half_return."""
        result = preprocess(sample_ohlcv_df, config)
        raw_std = result["half_return"].dropna().std()
        smooth_std = result["half_return_smooth"].dropna().std()
        assert smooth_std <= raw_std + 1e-10

    def test_high_vol_bar_creates_jump(self) -> None:
        """A bar with extreme return should be flagged as jump."""
        np.random.seed(42)
        n = 100
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        # Make one bar have extreme return: open=100, close=120
        close[50] = 120.0
        high = np.maximum(close, open_) + 1.0
        low = np.minimum(close, open_) - 1.0
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = HdMomRevConfig(jump_threshold=1.5)
        result = preprocess(df, config)
        # The extreme bar should have high jump_score
        assert result["jump_score"].iloc[50] > 0
