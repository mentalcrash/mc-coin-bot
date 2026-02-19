"""Tests for Liquidity-Confirmed Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.liq_conf_trend.config import LiqConfTrendConfig
from src.strategy.liq_conf_trend.preprocessor import preprocess


@pytest.fixture
def config() -> LiqConfTrendConfig:
    return LiqConfTrendConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def sample_full_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + all on-chain columns."""
    df = sample_ohlcv_df.copy()
    n = len(df)
    np.random.seed(42)
    df["oc_stablecoin_total_usd"] = 100e9 + np.cumsum(np.random.randn(n) * 1e8)
    df["oc_tvl_usd"] = 50e9 + np.cumsum(np.random.randn(n) * 5e7)
    df["oc_fear_greed"] = np.random.randint(10, 90, n).astype(float)
    return df


class TestPreprocess:
    def test_output_columns_ohlcv_only(
        self, sample_ohlcv_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        """OHLCV only: base columns + NaN for on-chain."""
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "price_mom",
            "stablecoin_roc",
            "tvl_roc",
            "liq_score",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_output_columns_full(
        self, sample_full_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        """Full data: all columns present."""
        result = preprocess(sample_full_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "price_mom",
            "stablecoin_roc",
            "tvl_roc",
            "liq_score",
            "fg_ma",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_full_df: pd.DataFrame, config: LiqConfTrendConfig) -> None:
        result = preprocess(sample_full_df, config)
        assert len(result) == len(sample_full_df)

    def test_immutability(self, sample_full_df: pd.DataFrame, config: LiqConfTrendConfig) -> None:
        original = sample_full_df.copy()
        preprocess(sample_full_df, config)
        pd.testing.assert_frame_equal(sample_full_df, original)

    def test_missing_columns(self, config: LiqConfTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_full_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        result = preprocess(sample_full_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_liq_score_range(
        self, sample_full_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        result = preprocess(sample_full_df, config)
        valid = result["liq_score"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 2).all()

    def test_graceful_degradation_no_onchain(
        self, sample_ohlcv_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        """Without on-chain data, liq_score should be 0."""
        result = preprocess(sample_ohlcv_df, config)
        assert (result["liq_score"] == 0).all()
        assert result["stablecoin_roc"].isna().all()
        assert result["tvl_roc"].isna().all()

    def test_drawdown_nonpositive(
        self, sample_full_df: pd.DataFrame, config: LiqConfTrendConfig
    ) -> None:
        result = preprocess(sample_full_df, config)
        valid_dd = result["drawdown"].dropna()
        assert (valid_dd <= 0).all()
