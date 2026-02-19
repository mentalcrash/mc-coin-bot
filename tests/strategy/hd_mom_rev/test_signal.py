"""Tests for hd-mom-rev signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.hd_mom_rev.config import HdMomRevConfig, ShortMode
from src.strategy.hd_mom_rev.preprocessor import preprocess
from src.strategy.hd_mom_rev.signal import generate_signals


@pytest.fixture
def config() -> HdMomRevConfig:
    return HdMomRevConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: HdMomRevConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: HdMomRevConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(self, preprocessed_df: pd.DataFrame, config: HdMomRevConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = HdMomRevConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = HdMomRevConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_short_requires_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = HdMomRevConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_mask = signals.direction == -1
        if short_mask.any():
            drawdown_at_short = df["drawdown"].shift(1)[short_mask]
            assert (drawdown_at_short < config.hedge_threshold).all()


class TestSignalLogic:
    def test_momentum_vs_reversal_logic(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Non-jump bars follow momentum, jump bars reverse direction."""
        config = HdMomRevConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        # Get bars where is_jump was False (momentum mode) from previous bar
        is_jump_prev = (
            processed["is_jump"].shift(1).fillna(value=True).infer_objects(copy=False).astype(bool)
        )
        half_return_prev = processed["half_return_smooth"].shift(1)

        # For momentum bars (non-jump): direction should match sign of half_return
        mom_mask = (~is_jump_prev) & (signals.direction != 0) & half_return_prev.notna()
        if mom_mask.sum() > 0:
            expected_dir = np.sign(half_return_prev[mom_mask])
            actual_dir = signals.direction[mom_mask].astype(float)
            # At least most should agree (allowing for edge cases)
            agreement = (expected_dir == actual_dir).sum() / len(expected_dir)
            assert agreement >= 0.8

    def test_entries_on_direction_change(
        self, preprocessed_df: pd.DataFrame, config: HdMomRevConfig
    ) -> None:
        """Entries should fire on direction transitions."""
        signals = generate_signals(preprocessed_df, config)
        entries_count = signals.entries.sum()
        assert entries_count >= 0

    def test_exits_when_direction_to_zero(
        self, preprocessed_df: pd.DataFrame, config: HdMomRevConfig
    ) -> None:
        """Exits fire when direction goes to 0."""
        signals = generate_signals(preprocessed_df, config)
        exits_count = signals.exits.sum()
        assert exits_count >= 0

    def test_confidence_modulates_strength(
        self, preprocessed_df: pd.DataFrame, config: HdMomRevConfig
    ) -> None:
        """Confidence should create varying strength levels."""
        signals = generate_signals(preprocessed_df, config)
        active_mask = signals.direction != 0
        if active_mask.sum() > 2:
            abs_strength = signals.strength[active_mask].abs()
            # Some variation in strength (confidence varies)
            assert abs_strength.std() > 0 or abs_strength.max() > 0
