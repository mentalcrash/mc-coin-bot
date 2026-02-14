"""Tests for fr-cond-mom preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.fr_cond_mom.config import FrCondMomConfig
from src.strategy.fr_cond_mom.preprocessor import preprocess


@pytest.fixture
def config() -> FrCondMomConfig:
    return FrCondMomConfig()


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
    funding_rate = np.random.uniform(-0.001, 0.001, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="6h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "mom_signal",
            "fr_ma",
            "fr_zscore",
            "fr_conviction",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: FrCondMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: FrCondMomConfig) -> None:
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "open": np.random.randn(n) + 100,
                "high": np.random.randn(n) + 101,
                "low": np.random.randn(n) + 99,
                "close": np.random.randn(n) + 100,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="6h"),
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_present(self, sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert "drawdown" in result.columns
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_mom_signal_values(
        self, sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig
    ) -> None:
        """mom_signal should be -1, 0, or 1."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["mom_signal"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_fr_conviction_range(
        self, sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig
    ) -> None:
        """fr_conviction should be between fr_dampening and 1.0."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["fr_conviction"].dropna()
        assert (valid >= config.fr_dampening - 1e-10).all()
        assert (valid <= 1.0 + 1e-10).all()

    def test_fr_zscore_computed(
        self, sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig
    ) -> None:
        """fr_zscore should have valid values after warmup."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["fr_zscore"].dropna()
        assert len(valid) > 0

    def test_fr_ma_smoothing(self, sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig) -> None:
        """fr_ma should be smoother than raw funding_rate."""
        result = preprocess(sample_ohlcv_df, config)
        raw_std = result["funding_rate"].dropna().std()
        ma_std = result["fr_ma"].dropna().std()
        assert ma_std <= raw_std
