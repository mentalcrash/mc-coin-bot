"""Tests for Carry-Regime Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.carry_regime_12h.config import CarryRegimeConfig
from src.strategy.carry_regime_12h.preprocessor import preprocess


@pytest.fixture
def config() -> CarryRegimeConfig:
    return CarryRegimeConfig()


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
def sample_ohlcv_with_fr_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + funding_rate."""
    np.random.seed(123)
    n = len(sample_ohlcv_df)
    df = sample_ohlcv_df.copy()
    df["funding_rate"] = np.random.uniform(-0.001, 0.001, n)
    return df


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "ema_fast",
            "ema_mid",
            "ema_slow",
            "ema_alignment",
            "fr_percentile",
            "drawdown",
            "atr",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: CarryRegimeConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_ema_alignment_range(
        self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["ema_alignment"].dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()

    def test_fr_percentile_graceful_degradation(
        self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        """funding_rate 없이도 fr_percentile=0.5으로 동작."""
        result = preprocess(sample_ohlcv_df, config)
        assert (result["fr_percentile"] == 0.5).all()

    def test_fr_percentile_with_funding_rate(
        self, sample_ohlcv_with_fr_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        """funding_rate 있으면 실제 percentile 계산."""
        result = preprocess(sample_ohlcv_with_fr_df, config)
        valid = result["fr_percentile"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()
        # Should not all be 0.5 when actual data is present
        assert not (valid == 0.5).all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
