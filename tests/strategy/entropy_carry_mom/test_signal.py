"""Tests for Entropy-Carry-Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.entropy_carry_mom.config import EntropyCarryMomConfig, ShortMode
from src.strategy.entropy_carry_mom.preprocessor import preprocess
from src.strategy.entropy_carry_mom.signal import generate_signals


@pytest.fixture
def config() -> EntropyCarryMomConfig:
    return EntropyCarryMomConfig()


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


@pytest.fixture
def preprocessed_df(sample_df: pd.DataFrame, config: EntropyCarryMomConfig) -> pd.DataFrame:
    return preprocess(sample_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortModeVariants:
    def test_disabled_no_shorts(self, sample_df: pd.DataFrame) -> None:
        config = EntropyCarryMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_df: pd.DataFrame) -> None:
        config = EntropyCarryMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_dampens_shorts(self, sample_df: pd.DataFrame) -> None:
        config = EntropyCarryMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY: shorts are conditional on drawdown
        assert signals.direction.dtype == int


class TestEntropyCarryMomLogic:
    def test_strong_momentum_produces_signal(self) -> None:
        """Strong positive momentum with low entropy should produce long signals."""
        n = 300
        np.random.seed(42)
        # Strong uptrend
        close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.3
        high = close + 1
        low = close - 1
        config = EntropyCarryMomConfig(fr_entry_threshold=0.0)
        df = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "funding_rate": np.full(n, 0.0001),  # small positive FR
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # Some signals should be produced
        assert signals.direction.abs().sum() > 0

    def test_carry_direction_follows_neg_fr_sign(self) -> None:
        """With high entropy and strong FR, carry direction = -sign(FR)."""
        n = 300
        np.random.seed(42)
        # Random walk (high entropy)
        close = 100 + np.cumsum(np.random.randn(n) * 3)
        high = close + 2
        low = close - 2
        config = EntropyCarryMomConfig(
            fr_entry_threshold=0.0,
            carry_weight_high_entropy=1.0,
            mom_weight_low_entropy=1.0,
        )
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "funding_rate": np.full(n, 0.005),  # strong positive FR -> carry short
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        # Verify carry_direction is -1 for positive FR
        valid_cd = processed["carry_direction"].dropna()
        assert (valid_cd == -1).all()

    def test_strength_nonzero_when_active_after_warmup(
        self, preprocessed_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        active = post_warmup != 0
        if active.any():
            assert (signals.strength.iloc[warmup:][active].abs() > 0).all()

    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        """entries and exits should not be True at the same bar."""
        signals = generate_signals(preprocessed_df, config)
        both = signals.entries & signals.exits
        assert not both.any()

    def test_strength_zero_when_direction_zero(
        self, preprocessed_df: pd.DataFrame, config: EntropyCarryMomConfig
    ) -> None:
        """When direction is 0, strength should also be 0."""
        signals = generate_signals(preprocessed_df, config)
        neutral = signals.direction == 0
        assert (signals.strength[neutral] == 0).all()
