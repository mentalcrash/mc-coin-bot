"""Tests for Carry-Momentum Convergence preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.carry_mom_convergence.config import CarryMomConvergenceConfig
from src.strategy.carry_mom_convergence.preprocessor import preprocess


@pytest.fixture
def config() -> CarryMomConvergenceConfig:
    return CarryMomConvergenceConfig()


@pytest.fixture
def sample_ohlcv_with_fr_df() -> pd.DataFrame:
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
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_with_fr_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_fr_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "price_mom",
            "ema_fast",
            "ema_slow",
            "trend_direction",
            "avg_funding_rate",
            "fr_zscore",
            "convergence_score",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_with_fr_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_fr_df, config)
        assert len(result) == len(sample_ohlcv_with_fr_df)

    def test_immutability(
        self, sample_ohlcv_with_fr_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        original = sample_ohlcv_with_fr_df.copy()
        preprocess(sample_ohlcv_with_fr_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_with_fr_df, original)

    def test_missing_columns(self, config: CarryMomConvergenceConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: CarryMomConvergenceConfig) -> None:
        df = pd.DataFrame(
            {
                "open": [1, 2, 3],
                "high": [2, 3, 4],
                "low": [0, 1, 2],
                "close": [1.5, 2.5, 3.5],
                "volume": [100, 200, 300],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_with_fr_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_fr_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_trend_direction_values(
        self, sample_ohlcv_with_fr_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_fr_df, config)
        assert set(result["trend_direction"].unique()).issubset({-1, 0, 1})

    def test_convergence_score_values(
        self, sample_ohlcv_with_fr_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_fr_df, config)
        valid = result["convergence_score"].dropna()
        expected_values = {config.convergence_boost, config.divergence_penalty, 1.0}
        assert set(valid.unique()).issubset(expected_values)

    def test_drawdown_non_positive(
        self, sample_ohlcv_with_fr_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_fr_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()
