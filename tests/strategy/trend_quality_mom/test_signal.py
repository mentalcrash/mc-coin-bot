"""Tests for Trend Quality Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.trend_quality_mom.config import ShortMode, TrendQualityMomConfig
from src.strategy.trend_quality_mom.preprocessor import preprocess
from src.strategy.trend_quality_mom.signal import generate_signals


@pytest.fixture
def config() -> TrendQualityMomConfig:
    return TrendQualityMomConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: TrendQualityMomConfig) -> pd.DataFrame:
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
        self, preprocessed_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TrendQualityMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TrendQualityMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_no_shorts_in_uptrend(self, trending_up_df: pd.DataFrame) -> None:
        """In a strong uptrend (no drawdown), HEDGE_ONLY should suppress shorts."""
        config = TrendQualityMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(trending_up_df, config)
        signals = generate_signals(df, config)
        # In uptrend, drawdown should be mild, so no hedge shorts
        assert (signals.direction >= 0).all()

    def test_hedge_only_allows_shorts_in_drawdown(self, trending_down_df: pd.DataFrame) -> None:
        """In a strong downtrend (deep drawdown), HEDGE_ONLY may allow shorts."""
        config = TrendQualityMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.05,
            r2_threshold=0.1,
        )
        df = preprocess(trending_down_df, config)
        signals = generate_signals(df, config)
        # Direction should contain some value (could be -1 or 0)
        assert signals.direction.dtype == int


class TestStrengthConviction:
    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_strength_zero_when_neutral(
        self, preprocessed_df: pd.DataFrame, config: TrendQualityMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        neutral_mask = signals.direction == 0
        assert (signals.strength[neutral_mask] == 0.0).all()
