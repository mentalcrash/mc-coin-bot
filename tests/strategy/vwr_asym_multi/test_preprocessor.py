"""Tests for VWR Asymmetric Multi-Scale preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vwr_asym_multi.config import VwrAsymMultiConfig
from src.strategy.vwr_asym_multi.preprocessor import preprocess


@pytest.fixture
def config() -> VwrAsymMultiConfig:
    return VwrAsymMultiConfig()


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
        self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "vwr_10",
            "vwr_21",
            "vwr_42",
            "vwr_zscore_10",
            "vwr_zscore_21",
            "vwr_zscore_42",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: VwrAsymMultiConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_vwr_columns_for_each_lookback(
        self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        for lb in (config.lookback_short, config.lookback_mid, config.lookback_long):
            assert f"vwr_{lb}" in result.columns
            assert f"vwr_zscore_{lb}" in result.columns

    def test_custom_lookbacks(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VwrAsymMultiConfig(lookback_short=5, lookback_mid=15, lookback_long=30)
        result = preprocess(sample_ohlcv_df, config)
        assert "vwr_5" in result.columns
        assert "vwr_15" in result.columns
        assert "vwr_30" in result.columns
        assert "vwr_zscore_5" in result.columns
        assert "vwr_zscore_15" in result.columns
        assert "vwr_zscore_30" in result.columns

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0).all()

    def test_returns_log(self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        # Log returns: first value is NaN
        assert pd.isna(result["returns"].iloc[0])
        # Finite for the rest
        valid = result["returns"].iloc[1:]
        assert np.isfinite(valid).all()

    def test_zscore_reasonable_range(
        self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        """Z-score는 대부분 [-5, 5] 범위 내."""
        result = preprocess(sample_ohlcv_df, config)
        for lb in (config.lookback_short, config.lookback_mid, config.lookback_long):
            valid = result[f"vwr_zscore_{lb}"].dropna()
            if len(valid) > 0:
                assert valid.abs().max() < 10, f"vwr_zscore_{lb} z-score out of range"


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
