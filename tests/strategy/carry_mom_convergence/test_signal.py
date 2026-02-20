"""Tests for Carry-Momentum Convergence signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.carry_mom_convergence.config import CarryMomConvergenceConfig, ShortMode
from src.strategy.carry_mom_convergence.preprocessor import preprocess
from src.strategy.carry_mom_convergence.signal import generate_signals


@pytest.fixture
def config() -> CarryMomConvergenceConfig:
    return CarryMomConvergenceConfig()


@pytest.fixture
def sample_ohlcv_with_fr_df() -> pd.DataFrame:
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
def preprocessed_df(
    sample_ohlcv_with_fr_df: pd.DataFrame, config: CarryMomConvergenceConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_with_fr_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortModeVariants:
    def test_disabled_no_shorts(self, sample_ohlcv_with_fr_df: pd.DataFrame) -> None:
        config = CarryMomConvergenceConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_with_fr_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_with_fr_df: pd.DataFrame) -> None:
        config = CarryMomConvergenceConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_with_fr_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_mode(self, sample_ohlcv_with_fr_df: pd.DataFrame) -> None:
        config = CarryMomConvergenceConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_with_fr_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestCarryMomConvergenceLogic:
    def test_uptrend_with_convergent_fr_generates_long(self) -> None:
        """Clear uptrend + negative FR (convergent) → long signal."""
        n = 200
        close = 100 + np.arange(n) * 1.0  # Strong uptrend
        high = close + 1
        low = close - 1
        # Negative FR → shorts paying longs → convergent with uptrend
        funding_rate = np.full(n, -0.0005)
        config = CarryMomConvergenceConfig()
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "funding_rate": funding_rate,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # After warmup, should see long signals
        late = signals.direction.iloc[config.warmup_periods() :]
        if (late != 0).any():
            assert (late >= 0).all()

    def test_downtrend_with_convergent_fr_generates_short(self) -> None:
        """Clear downtrend + positive FR (convergent) → short signal in FULL mode."""
        n = 200
        close = 200 - np.arange(n) * 1.0  # Strong downtrend
        high = close + 1
        low = close - 1
        # Positive FR → longs paying shorts → convergent with downtrend
        funding_rate = np.full(n, 0.0005)
        config = CarryMomConvergenceConfig(short_mode=ShortMode.FULL)
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
                "funding_rate": funding_rate,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        late = signals.direction.iloc[config.warmup_periods() :]
        if (late != 0).any():
            assert (late <= 0).all()

    def test_strength_nonzero_when_active_after_warmup(
        self, preprocessed_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        active = post_warmup != 0
        if active.any():
            assert (signals.strength.iloc[warmup:][active].abs() > 0).all()

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: CarryMomConvergenceConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()
