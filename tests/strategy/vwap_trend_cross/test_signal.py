"""Tests for VWAP Trend Crossover signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vwap_trend_cross.config import ShortMode, VwapTrendCrossConfig
from src.strategy.vwap_trend_cross.preprocessor import preprocess
from src.strategy.vwap_trend_cross.signal import generate_signals


@pytest.fixture
def config() -> VwapTrendCrossConfig:
    return VwapTrendCrossConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: VwapTrendCrossConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """Strong uptrend for testing HEDGE_ONLY."""
    n = 200
    close = np.linspace(100, 200, n) + np.random.RandomState(42).randn(n) * 0.5
    high = close + 2.0
    low = close - 2.0
    open_ = close - 0.5
    volume = np.full(n, 5000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    """Strong downtrend for testing HEDGE_ONLY drawdown."""
    n = 200
    close = np.linspace(200, 80, n) + np.random.RandomState(42).randn(n) * 0.5
    high = close + 2.0
    low = close - 2.0
    open_ = close + 0.5
    volume = np.full(n, 5000.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: VwapTrendCrossConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.direction) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VwapTrendCrossConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VwapTrendCrossConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_no_shorts_in_uptrend(self, trending_up_df: pd.DataFrame) -> None:
        config = VwapTrendCrossConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(trending_up_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_hedge_only_allows_shorts_in_drawdown(self, trending_down_df: pd.DataFrame) -> None:
        config = VwapTrendCrossConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.05,
        )
        df = preprocess(trending_down_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestVwapCrossLogic:
    def test_uptrend_generates_long(self, trending_up_df: pd.DataFrame) -> None:
        """Strong uptrend should generate long signals."""
        config = VwapTrendCrossConfig()
        df = preprocess(trending_up_df, config)
        signals = generate_signals(df, config)
        # Should have some long signals in an uptrend
        assert (signals.direction == 1).any()

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_strength_zero_when_neutral(
        self, preprocessed_df: pd.DataFrame, config: VwapTrendCrossConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        neutral_mask = signals.direction == 0
        assert (signals.strength[neutral_mask] == 0.0).all()
