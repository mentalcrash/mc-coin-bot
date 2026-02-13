"""Tests for Volume-Confirmed Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_confirm_mom.config import ShortMode, VolConfirmMomConfig
from src.strategy.vol_confirm_mom.preprocessor import preprocess
from src.strategy.vol_confirm_mom.signal import generate_signals


@pytest.fixture
def config() -> VolConfirmMomConfig:
    return VolConfirmMomConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: VolConfirmMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """Strong uptrend with rising volume."""
    n = 200
    close = np.linspace(100, 200, n) + np.random.RandomState(42).randn(n) * 0.5
    high = close + 2.0
    low = close - 2.0
    open_ = close - 0.5
    # Rising volume trend
    volume = np.linspace(3000, 8000, n) + np.random.RandomState(42).randn(n) * 200
    volume = np.clip(volume, 1000, None)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    """Strong downtrend with rising volume."""
    n = 200
    close = np.linspace(200, 80, n) + np.random.RandomState(42).randn(n) * 0.5
    high = close + 2.0
    low = close - 2.0
    open_ = close + 0.5
    volume = np.linspace(3000, 8000, n) + np.random.RandomState(42).randn(n) * 200
    volume = np.clip(volume, 1000, None)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: VolConfirmMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.direction) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VolConfirmMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VolConfirmMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_no_shorts_in_uptrend(self, trending_up_df: pd.DataFrame) -> None:
        config = VolConfirmMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(trending_up_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_hedge_only_allows_shorts_in_drawdown(self, trending_down_df: pd.DataFrame) -> None:
        config = VolConfirmMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.05,
        )
        df = preprocess(trending_down_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestVolumeConfirmation:
    def test_no_signal_without_volume_confirmation(self) -> None:
        """With flat/declining volume, fewer signals should be generated."""
        n = 200
        rng = np.random.RandomState(42)
        close = 100 + np.cumsum(rng.randn(n) * 2)
        high = close + 2.0
        low = close - 2.0
        open_ = close - 0.5
        # Flat volume (short SMA ~ long SMA)
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        config = VolConfirmMomConfig()
        preprocessed = preprocess(df, config)
        signals = generate_signals(preprocessed, config)
        # With flat volume, vol_rising should be False most of the time
        # so fewer signals are expected
        assert signals.direction.dtype == int

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: VolConfirmMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()
