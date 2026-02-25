"""Tests for Donchian Multi-Scale preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.donch_multi.config import DonchMultiConfig
from src.strategy.donch_multi.preprocessor import preprocess


@pytest.fixture
def config() -> DonchMultiConfig:
    return DonchMultiConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "drawdown",
            "dc_upper_20",
            "dc_lower_20",
            "dc_upper_40",
            "dc_lower_40",
            "dc_upper_80",
            "dc_lower_80",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: DonchMultiConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_donchian_channels_valid(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        """Donchian upper >= lower for all scales."""
        result = preprocess(sample_ohlcv_df, config)
        for lb in (config.lookback_short, config.lookback_mid, config.lookback_long):
            upper = result[f"dc_upper_{lb}"].dropna()
            lower = result[f"dc_lower_{lb}"].dropna()
            assert (upper >= lower).all()

    def test_custom_lookbacks(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """커스텀 lookback으로 컬럼 생성 확인."""
        config = DonchMultiConfig(lookback_short=10, lookback_mid=30, lookback_long=60)
        result = preprocess(sample_ohlcv_df, config)
        assert "dc_upper_10" in result.columns
        assert "dc_lower_30" in result.columns
        assert "dc_upper_60" in result.columns

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0.0).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
