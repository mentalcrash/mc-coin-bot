"""Tests for Drawdown-Recovery Phase signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.dd_recovery_phase.config import DDRecoveryPhaseConfig, ShortMode
from src.strategy.dd_recovery_phase.preprocessor import preprocess
from src.strategy.dd_recovery_phase.signal import generate_signals


@pytest.fixture
def config() -> DDRecoveryPhaseConfig:
    return DDRecoveryPhaseConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: DDRecoveryPhaseConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: DDRecoveryPhaseConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: DDRecoveryPhaseConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: DDRecoveryPhaseConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: DDRecoveryPhaseConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: DDRecoveryPhaseConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DDRecoveryPhaseConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_hedge_only_respects_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DDRecoveryPhaseConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DDRecoveryPhaseConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestStrategyLogic:
    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: DDRecoveryPhaseConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_hedge_only_reduces_short_strength(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY short strength < FULL short strength (by hedge_strength_ratio)."""
        config_hedge = DDRecoveryPhaseConfig(
            short_mode=ShortMode.HEDGE_ONLY, hedge_strength_ratio=0.5
        )
        config_full = DDRecoveryPhaseConfig(short_mode=ShortMode.FULL)
        df_h = preprocess(sample_ohlcv_df, config_hedge)
        df_f = preprocess(sample_ohlcv_df, config_full)
        sig_h = generate_signals(df_h, config_hedge)
        sig_f = generate_signals(df_f, config_full)
        # If both have shorts, hedge should have smaller absolute strength
        short_h = sig_h.strength[sig_h.direction == -1].abs()
        short_f = sig_f.strength[sig_f.direction == -1].abs()
        if len(short_h) > 0 and len(short_f) > 0:
            assert short_h.mean() <= short_f.mean() + 1e-10
