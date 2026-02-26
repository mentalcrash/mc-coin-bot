"""Tests for Donchian Cascade MTF preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.donch_cascade.config import DonchCascadeConfig
from src.strategy.donch_cascade.preprocessor import preprocess


@pytest.fixture
def config() -> DonchCascadeConfig:
    return DonchCascadeConfig()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 800  # 4H bars (enough for 240 lookback warmup)
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocessOutput:
    def test_donchian_columns_exist(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        df = preprocess(sample_ohlcv_df, config)
        actual_lbs = config.actual_lookbacks()
        for lb in actual_lbs:
            assert f"dc_upper_{lb}" in df.columns
            assert f"dc_lower_{lb}" in df.columns

    def test_confirm_ema_exists(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        df = preprocess(sample_ohlcv_df, config)
        assert "confirm_ema" in df.columns

    def test_vol_columns_exist(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        df = preprocess(sample_ohlcv_df, config)
        assert "returns" in df.columns
        assert "realized_vol" in df.columns
        assert "vol_scalar" in df.columns

    def test_drawdown_exists(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        df = preprocess(sample_ohlcv_df, config)
        assert "drawdown" in df.columns

    def test_original_columns_preserved(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        df = preprocess(sample_ohlcv_df, config)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig) -> None:
        df = preprocess(sample_ohlcv_df, config)
        assert len(df) == len(sample_ohlcv_df)


class TestActualLookbacks:
    def test_lookback_uses_multiplier(self) -> None:
        """actual lookback = base * htf_multiplier."""
        config = DonchCascadeConfig(lookback_short=10, lookback_mid=20, lookback_long=30)
        actual_lbs = config.actual_lookbacks()
        assert actual_lbs == (30, 60, 90)

    def test_donchian_channels_use_actual_lookback(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Donchian channel은 actual lookback으로 계산."""
        config = DonchCascadeConfig(
            lookback_short=10, lookback_mid=20, lookback_long=30, htf_multiplier=2
        )
        df = preprocess(sample_ohlcv_df, config)
        # actual lookback = 20, 40, 60
        assert "dc_upper_20" in df.columns
        assert "dc_upper_40" in df.columns
        assert "dc_upper_60" in df.columns


class TestDoesNotMutateInput:
    def test_original_df_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        original_cols = set(sample_ohlcv_df.columns)
        preprocess(sample_ohlcv_df, config)
        assert set(sample_ohlcv_df.columns) == original_cols


class TestMissingColumns:
    def test_missing_close(self, config: DonchCascadeConfig) -> None:
        df = pd.DataFrame(
            {"open": [1.0], "high": [2.0], "low": [0.5], "volume": [100.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="4h"),
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_empty_df(self, config: DonchCascadeConfig) -> None:
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
        )
        df.index = pd.DatetimeIndex([], name="timestamp")
        # Empty DF raises ValueError from log_returns (expected)
        with pytest.raises(ValueError, match="Empty Series"):
            preprocess(df, config)
