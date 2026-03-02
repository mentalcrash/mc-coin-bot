"""Tests for R2 Consensus Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.r2_consensus.config import R2ConsensusConfig
from src.strategy.r2_consensus.preprocessor import preprocess


@pytest.fixture
def config() -> R2ConsensusConfig:
    return R2ConsensusConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "slope_20",
            "r2_20",
            "slope_50",
            "r2_50",
            "slope_120",
            "r2_120",
            "atr",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: R2ConsensusConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_r2_range_0_to_1(
        self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        """R^2 값이 0~1 범위 내에 있어야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        for lb in (config.lookback_short, config.lookback_mid, config.lookback_long):
            r2 = result[f"r2_{lb}"].dropna()
            assert (r2 >= 0.0).all(), f"r2_{lb} has values < 0"
            assert (r2 <= 1.0).all(), f"r2_{lb} has values > 1"

    def test_three_scales_present(
        self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        """3개 스케일의 slope/r2 컬럼이 모두 존재."""
        result = preprocess(sample_ohlcv_df, config)
        for lb in (20, 50, 120):
            assert f"slope_{lb}" in result.columns
            assert f"r2_{lb}" in result.columns

    def test_custom_lookbacks(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """커스텀 lookback에 맞는 컬럼명 생성."""
        config = R2ConsensusConfig(lookback_short=10, lookback_mid=40, lookback_long=80)
        result = preprocess(sample_ohlcv_df, config)
        assert "slope_10" in result.columns
        assert "r2_40" in result.columns
        assert "slope_80" in result.columns

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        """drawdown은 0 이하여야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0.0).all()

    def test_returns_log(self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig) -> None:
        """log return 계산 검증."""
        result = preprocess(sample_ohlcv_df, config)
        expected = np.log(sample_ohlcv_df["close"] / sample_ohlcv_df["close"].shift(1))
        pd.testing.assert_series_equal(result["returns"], expected, check_names=False)


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
