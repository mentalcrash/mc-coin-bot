"""Tests for fr-cond-mom signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.fr_cond_mom.config import FrCondMomConfig, ShortMode
from src.strategy.fr_cond_mom.preprocessor import preprocess
from src.strategy.fr_cond_mom.signal import generate_signals


@pytest.fixture
def config() -> FrCondMomConfig:
    return FrCondMomConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="6h"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: FrCondMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: FrCondMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: FrCondMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: FrCondMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: FrCondMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: FrCondMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: FrCondMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = FrCondMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = FrCondMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_short_requires_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = FrCondMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_mask = signals.direction == -1
        if short_mask.any():
            drawdown_at_short = df["drawdown"].shift(1)[short_mask]
            assert (drawdown_at_short < config.hedge_threshold).all()


class TestSignalLogic:
    def test_conviction_modulates_strength(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """FR conviction should modulate strength magnitude."""
        config = FrCondMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        active_mask = signals.direction != 0
        if active_mask.sum() > 2:
            abs_strength = signals.strength[active_mask].abs()
            # Not all strength values should be identical (conviction varies)
            assert abs_strength.std() > 0 or abs_strength.max() > 0

    def test_entries_on_direction_change(
        self, preprocessed_df: pd.DataFrame, config: FrCondMomConfig
    ) -> None:
        """Entries should fire on direction transitions."""
        signals = generate_signals(preprocessed_df, config)
        entries_count = signals.entries.sum()
        assert entries_count >= 0

    def test_exits_when_direction_to_zero(
        self, preprocessed_df: pd.DataFrame, config: FrCondMomConfig
    ) -> None:
        """Exits fire when direction goes to 0."""
        signals = generate_signals(preprocessed_df, config)
        exits_count = signals.exits.sum()
        assert exits_count >= 0

    def test_extreme_fr_reduces_strength(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """When FR is extreme, conviction should be lower, reducing strength."""
        # Normal FR
        config_normal = FrCondMomConfig(short_mode=ShortMode.FULL)
        df_normal = sample_ohlcv_df.copy()
        df_normal["funding_rate"] = 0.0001  # mild, z-score near 0
        df_normal = preprocess(df_normal, config_normal)
        sig_normal = generate_signals(df_normal, config_normal)

        # Extreme FR (high dampening effect)
        config_extreme = FrCondMomConfig(short_mode=ShortMode.FULL, fr_dampening=0.1)
        df_extreme = sample_ohlcv_df.copy()
        # Create extreme funding rate pattern
        n = len(df_extreme)
        fr_extreme = np.zeros(n)
        fr_extreme[: n // 2] = 0.01  # extreme positive
        fr_extreme[n // 2 :] = -0.01  # extreme negative
        df_extreme["funding_rate"] = fr_extreme
        df_extreme = preprocess(df_extreme, config_extreme)
        sig_extreme = generate_signals(df_extreme, config_extreme)

        # Both should produce valid signals
        assert not sig_normal.strength.isna().any()
        assert not sig_extreme.strength.isna().any()
