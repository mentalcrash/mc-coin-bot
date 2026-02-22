"""Tests for Hash-Ribbon Capitulation preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.hash_ribbon_cap.config import HashRibbonCapConfig
from src.strategy.hash_ribbon_cap.preprocessor import preprocess


@pytest.fixture
def config() -> HashRibbonCapConfig:
    return HashRibbonCapConfig()


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


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "sma_fast",
            "sma_slow",
            "capitulation",
            "recovery_bars",
            "had_capitulation",
            "momentum",
            "drawdown",
            "atr",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: HashRibbonCapConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: HashRibbonCapConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: HashRibbonCapConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_capitulation_binary(
        self, sample_ohlcv_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["capitulation"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_recovery_bars_non_negative(
        self, sample_ohlcv_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert (result["recovery_bars"] >= 0).all()

    def test_momentum_values(
        self, sample_ohlcv_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["momentum"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})
