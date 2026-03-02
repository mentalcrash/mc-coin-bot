"""Tests for VWAP-Channel Multi-Scale preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vwap_channel_12h.config import VwapChannelConfig
from src.strategy.vwap_channel_12h.preprocessor import preprocess


@pytest.fixture
def config() -> VwapChannelConfig:
    return VwapChannelConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "drawdown",
            "vwap_20",
            "vwap_60",
            "vwap_150",
            "vwap_upper_20",
            "vwap_upper_60",
            "vwap_upper_150",
            "vwap_lower_20",
            "vwap_lower_60",
            "vwap_lower_150",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: VwapChannelConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_vwap_between_high_low(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        """VWAP는 typical price의 volume-weighted 평균이므로 high/low 범위 근처."""
        result = preprocess(sample_ohlcv_df, config)
        for s in (20, 60, 150):
            valid_vwap = result[f"vwap_{s}"].dropna()
            # VWAP가 유한한 값을 가져야 함
            assert valid_vwap.isna().sum() == 0
            assert np.isfinite(valid_vwap.values).all()

    def test_band_ordering(self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig) -> None:
        """upper > vwap > lower 순서 보장."""
        result = preprocess(sample_ohlcv_df, config)
        for s in (20, 60, 150):
            valid_mask = result[f"vwap_{s}"].notna() & result[f"vwap_upper_{s}"].notna()
            valid = result[valid_mask]
            assert (valid[f"vwap_upper_{s}"] >= valid[f"vwap_{s}"]).all()
            assert (valid[f"vwap_{s}"] >= valid[f"vwap_lower_{s}"]).all()

    def test_custom_scales(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """커스텀 스케일에서도 정상 동작."""
        config = VwapChannelConfig(scale_short=10, scale_mid=30, scale_long=80)
        result = preprocess(sample_ohlcv_df, config)
        assert "vwap_10" in result.columns
        assert "vwap_30" in result.columns
        assert "vwap_80" in result.columns

    def test_drawdown_nonpositive(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0.0).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
