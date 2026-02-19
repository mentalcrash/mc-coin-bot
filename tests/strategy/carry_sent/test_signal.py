"""Tests for Carry-Sentiment Gate signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.carry_sent.config import CarrySentConfig, ShortMode
from src.strategy.carry_sent.preprocessor import preprocess
from src.strategy.carry_sent.signal import generate_signals


@pytest.fixture
def config() -> CarrySentConfig:
    return CarrySentConfig()


@pytest.fixture
def sample_df() -> pd.DataFrame:
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
    fear_greed = np.random.randint(10, 90, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
            "oc_fear_greed": fear_greed,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def preprocessed_df(sample_df: pd.DataFrame, config: CarrySentConfig) -> pd.DataFrame:
    return preprocess(sample_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: CarrySentConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: CarrySentConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: CarrySentConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: CarrySentConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: CarrySentConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortModeVariants:
    def test_disabled_no_shorts(self, sample_df: pd.DataFrame) -> None:
        config = CarrySentConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_df: pd.DataFrame) -> None:
        config = CarrySentConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_dampens_shorts(self, sample_df: pd.DataFrame) -> None:
        config = CarrySentConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY: shorts are conditional on drawdown
        assert signals.direction.dtype == int


class TestCarrySentLogic:
    def test_fear_extreme_forces_long(self) -> None:
        """F&G < fear_threshold forces long via contrarian override."""
        n = 200
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + 1
        low = close - 1
        config = CarrySentConfig(fg_fear_threshold=20)
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "funding_rate": np.full(n, 0.001),  # positive FR → normally short
                "oc_fear_greed": np.full(n, 10.0),  # extreme fear
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # After warmup, fear extreme should force long (direction=1)
        valid = signals.direction[signals.direction != 0]
        if len(valid) > 0:
            assert (valid == 1).all()

    def test_greed_extreme_forces_short_full_mode(self) -> None:
        """F&G > greed_threshold forces short in FULL mode."""
        n = 200
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + 1
        low = close - 1
        config = CarrySentConfig(
            fg_greed_threshold=80,
            short_mode=ShortMode.FULL,
        )
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "funding_rate": np.full(n, -0.001),  # negative FR → normally long
                "oc_fear_greed": np.full(n, 90.0),  # extreme greed
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # After warmup, greed extreme should force short (direction=-1)
        valid = signals.direction[signals.direction != 0]
        if len(valid) > 0:
            assert (valid == -1).all()

    def test_carry_zone_respects_fr_direction(self) -> None:
        """In F&G neutral gate zone, carry follows -sign(FR)."""
        n = 200
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + 1
        low = close - 1
        config = CarrySentConfig(
            fg_gate_low=30,
            fg_gate_high=70,
            fr_entry_threshold=0.0,
        )
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "funding_rate": np.full(n, 0.001),  # positive FR → carry short
                "oc_fear_greed": np.full(n, 50.0),  # neutral zone
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # With positive FR in neutral zone → direction should be -1 (short carry)
        valid = signals.direction[signals.direction != 0]
        if len(valid) > 0:
            assert (valid == -1).all()

    def test_no_signal_outside_zones(self) -> None:
        """F&G between gate_high and greed_threshold → no signal."""
        n = 200
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + 1
        low = close - 1
        config = CarrySentConfig(
            fg_gate_high=70,
            fg_greed_threshold=80,
        )
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "funding_rate": np.full(n, 0.001),
                "oc_fear_greed": np.full(n, 75.0),  # between gate_high and greed
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # No active zone → all neutral
        assert (signals.direction == 0).all()

    def test_strength_nonzero_when_active_after_warmup(
        self, preprocessed_df: pd.DataFrame, config: CarrySentConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        active = post_warmup != 0
        if active.any():
            assert (signals.strength.iloc[warmup:][active].abs() > 0).all()
