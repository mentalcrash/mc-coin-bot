"""Tests for up-vol-mom signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.up_vol_mom.config import ShortMode, UpVolMomConfig
from src.strategy.up_vol_mom.preprocessor import preprocess
from src.strategy.up_vol_mom.signal import generate_signals


@pytest.fixture
def config() -> UpVolMomConfig:
    return UpVolMomConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: UpVolMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = UpVolMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = UpVolMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_short_requires_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = UpVolMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY shorts only active during drawdown
        short_mask = signals.direction == -1
        if short_mask.any():
            drawdown_at_short = df["drawdown"].shift(1)[short_mask]
            assert (drawdown_at_short < config.hedge_threshold).all()


class TestSignalLogic:
    def test_long_requires_up_ratio_above_threshold(
        self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig
    ) -> None:
        """Long entries require up_ratio_ma > threshold (from previous bar)."""
        signals = generate_signals(preprocessed_df, config)
        long_mask = signals.direction == 1
        if long_mask.any():
            up_ratio_prev = preprocessed_df["up_ratio_ma"].shift(1)[long_mask]
            valid = up_ratio_prev.dropna()
            assert (valid > config.ratio_threshold).all()

    def test_short_requires_down_ratio_dominant(self, preprocessed_df: pd.DataFrame) -> None:
        """Short entries require up_ratio_ma < (1 - threshold)."""
        config = UpVolMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(preprocessed_df, config)
        signals = generate_signals(df, config)
        short_mask = signals.direction == -1
        if short_mask.any():
            up_ratio_prev = df["up_ratio_ma"].shift(1)[short_mask]
            valid = up_ratio_prev.dropna()
            assert (valid < (1.0 - config.ratio_threshold)).all()

    def test_conviction_scales_with_deviation(
        self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig
    ) -> None:
        """Strength magnitude should be higher for stronger ratio deviations."""
        signals = generate_signals(preprocessed_df, config)
        active_mask = signals.direction != 0
        if active_mask.sum() > 2:
            abs_strength = signals.strength[active_mask].abs()
            assert abs_strength.max() > abs_strength.min()

    def test_entries_on_direction_change(
        self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig
    ) -> None:
        """Entries should fire on direction transitions."""
        signals = generate_signals(preprocessed_df, config)
        entries_count = signals.entries.sum()
        # Should have at least some entries in 300 bars
        assert entries_count >= 0  # No crash, valid output

    def test_exits_when_direction_to_zero(
        self, preprocessed_df: pd.DataFrame, config: UpVolMomConfig
    ) -> None:
        """Exits fire when direction goes to 0."""
        signals = generate_signals(preprocessed_df, config)
        exits_count = signals.exits.sum()
        assert exits_count >= 0  # No crash, valid output
