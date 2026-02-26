"""Tests for Triple-Channel Multi-Scale Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.tri_channel_trend.config import TriChannelTrendConfig
from src.strategy.tri_channel_trend.preprocessor import preprocess


@pytest.fixture
def config() -> TriChannelTrendConfig:
    return TriChannelTrendConfig()


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
        self, sample_ohlcv_df: pd.DataFrame, config: TriChannelTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        # 기본 feature
        required = {"returns", "realized_vol", "vol_scalar", "drawdown"}
        assert required.issubset(set(result.columns))
        # 채널 feature (3 channels x 3 scales = 18 columns)
        for s in (config.scale_short, config.scale_mid, config.scale_long):
            assert f"dc_upper_{s}" in result.columns
            assert f"dc_lower_{s}" in result.columns
            assert f"kc_upper_{s}" in result.columns
            assert f"kc_lower_{s}" in result.columns
            assert f"bb_upper_{s}" in result.columns
            assert f"bb_lower_{s}" in result.columns

    def test_same_length(
        self, sample_ohlcv_df: pd.DataFrame, config: TriChannelTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: TriChannelTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: TriChannelTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_nonpositive(
        self, sample_ohlcv_df: pd.DataFrame, config: TriChannelTrendConfig
    ) -> None:
        """Drawdown은 항상 <= 0."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_donchian_upper_gte_lower(
        self, sample_ohlcv_df: pd.DataFrame, config: TriChannelTrendConfig
    ) -> None:
        """Donchian upper >= lower."""
        result = preprocess(sample_ohlcv_df, config)
        for s in (config.scale_short, config.scale_mid, config.scale_long):
            upper = result[f"dc_upper_{s}"].dropna()
            lower = result[f"dc_lower_{s}"].dropna()
            assert (upper >= lower).all()

    def test_keltner_upper_gte_lower(
        self, sample_ohlcv_df: pd.DataFrame, config: TriChannelTrendConfig
    ) -> None:
        """Keltner upper >= lower."""
        result = preprocess(sample_ohlcv_df, config)
        for s in (config.scale_short, config.scale_mid, config.scale_long):
            upper = result[f"kc_upper_{s}"].dropna()
            lower = result[f"kc_lower_{s}"].dropna()
            assert (upper >= lower).all()

    def test_bb_upper_gte_lower(
        self, sample_ohlcv_df: pd.DataFrame, config: TriChannelTrendConfig
    ) -> None:
        """BB upper >= lower."""
        result = preprocess(sample_ohlcv_df, config)
        for s in (config.scale_short, config.scale_mid, config.scale_long):
            upper = result[f"bb_upper_{s}"].dropna()
            lower = result[f"bb_lower_{s}"].dropna()
            assert (upper >= lower).all()

    def test_custom_config(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """커스텀 config로도 정상 작동."""
        config = TriChannelTrendConfig(
            scale_short=10,
            scale_mid=30,
            scale_long=80,
            bb_std_dev=1.5,
            keltner_multiplier=2.0,
        )
        result = preprocess(sample_ohlcv_df, config)
        assert f"dc_upper_{config.scale_short}" in result.columns


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: TriChannelTrendConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: TriChannelTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
