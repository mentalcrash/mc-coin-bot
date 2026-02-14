"""Tests for Carry-Conditional Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.carry_cond_mom.config import CarryCondMomConfig
from src.strategy.carry_cond_mom.preprocessor import preprocess


@pytest.fixture
def config() -> CarryCondMomConfig:
    return CarryCondMomConfig()


@pytest.fixture
def sample_ohlcv_with_funding_df() -> pd.DataFrame:
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
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "price_mom",
            "avg_funding_rate",
            "fr_zscore",
            "mom_direction",
            "fr_direction",
            "agreement",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        assert len(result) == len(sample_ohlcv_with_funding_df)

    def test_immutability(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        original = sample_ohlcv_with_funding_df.copy()
        preprocess(sample_ohlcv_with_funding_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_with_funding_df, original)

    def test_missing_columns(self, config: CarryCondMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: CarryCondMomConfig) -> None:
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
                "volume": [100.0, 200.0, 300.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()


class TestAgreementFeatures:
    def test_direction_values(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid_mom = result["mom_direction"].dropna()
        valid_fr = result["fr_direction"].dropna()
        assert set(valid_mom.unique()).issubset({-1.0, 0.0, 1.0})
        assert set(valid_fr.unique()).issubset({-1.0, 0.0, 1.0})

    def test_agreement_values(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid = result["agreement"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_funding_rate_nan_ffill(self, config: CarryCondMomConfig) -> None:
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + 2
        low = close - 2
        open_ = close - 0.5
        volume = np.full(n, 5000.0)
        fr = np.random.uniform(-0.001, 0.001, n)
        fr[0:10] = np.nan
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "funding_rate": fr,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        result = preprocess(df, config)
        assert len(result) == n
