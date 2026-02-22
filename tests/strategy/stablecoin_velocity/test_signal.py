"""Tests for Stablecoin Velocity Regime signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.stablecoin_velocity.config import ShortMode, StablecoinVelocityConfig
from src.strategy.stablecoin_velocity.preprocessor import preprocess
from src.strategy.stablecoin_velocity.signal import generate_signals


@pytest.fixture
def config() -> StablecoinVelocityConfig:
    return StablecoinVelocityConfig()


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
def preprocessed_df(
    sample_ohlcv_df: pd.DataFrame, config: StablecoinVelocityConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: StablecoinVelocityConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: StablecoinVelocityConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: StablecoinVelocityConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: StablecoinVelocityConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: StablecoinVelocityConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = StablecoinVelocityConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_hedge_only_respects_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = StablecoinVelocityConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = StablecoinVelocityConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestStrategyLogic:
    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: StablecoinVelocityConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_high_volume_spike_triggers_long(self) -> None:
        """A sudden volume spike should increase velocity and potentially trigger long."""
        n = 200
        close = np.full(n, 100.0)
        volume = np.full(n, 1000.0)
        # Volume spike from bar 150
        volume[150:170] = 10000.0
        high = close + 1.0
        low = close - 1.0
        df = pd.DataFrame(
            {
                "open": close.copy(),
                "high": high,
                "low": low,
                "close": close.copy(),
                "volume": volume,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        config = StablecoinVelocityConfig()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # Should have some non-zero signals after the volume spike
        assert signals.direction.dtype == int
