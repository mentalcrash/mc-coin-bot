"""Tests for Donchian Filtered preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.donch_filtered.config import DonchFilteredConfig
from src.strategy.donch_filtered.preprocessor import preprocess


@pytest.fixture
def config() -> DonchFilteredConfig:
    return DonchFilteredConfig()


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


@pytest.fixture
def sample_ohlcv_with_fr(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + funding_rate 컬럼."""
    np.random.seed(123)
    df = sample_ohlcv_df.copy()
    df["funding_rate"] = np.random.randn(len(df)) * 0.001
    return df


class TestPreprocess:
    def test_output_columns_no_derivatives(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
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
            "fr_zscore",
        }
        assert required.issubset(set(result.columns))

    def test_output_columns_with_derivatives(
        self, sample_ohlcv_with_fr: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_fr, config)
        assert "fr_zscore" in result.columns

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: DonchFilteredConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_donchian_channels_valid(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        for lb in (config.lookback_short, config.lookback_mid, config.lookback_long):
            upper = result[f"dc_upper_{lb}"].dropna()
            lower = result[f"dc_lower_{lb}"].dropna()
            assert (upper >= lower).all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0.0).all()


class TestGracefulDegradation:
    def test_no_derivatives_fr_zscore_zero(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        """Derivatives 데이터 없으면 fr_zscore = 0.0."""
        result = preprocess(sample_ohlcv_df, config)
        assert (result["fr_zscore"] == 0.0).all()

    def test_with_derivatives_fr_zscore_computed(
        self, sample_ohlcv_with_fr: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        """Derivatives 데이터 있으면 fr_zscore가 실제 계산됨."""
        result = preprocess(sample_ohlcv_with_fr, config)
        # warmup 이후 NaN 아닌 값이 존재해야 함
        valid = result["fr_zscore"].dropna()
        assert len(valid) > 0
        # 모든 값이 0.0은 아님 (실제 계산됨)
        assert not (valid == 0.0).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_original_with_fr_unchanged(
        self, sample_ohlcv_with_fr: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        original = sample_ohlcv_with_fr.copy()
        preprocess(sample_ohlcv_with_fr, config)
        pd.testing.assert_frame_equal(sample_ohlcv_with_fr, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
