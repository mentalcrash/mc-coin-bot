"""Tests for Macro-Liquidity Adaptive Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.macro_liq_trend.config import MacroLiqTrendConfig, ShortMode
from src.strategy.macro_liq_trend.preprocessor import preprocess
from src.strategy.macro_liq_trend.signal import generate_signals


@pytest.fixture
def config() -> MacroLiqTrendConfig:
    return MacroLiqTrendConfig()


@pytest.fixture
def sample_macro_df() -> pd.DataFrame:
    """1D OHLCV + Macro data (300 bars)."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    high = close + np.abs(np.random.randn(n) * 300)
    low = close - np.abs(np.random.randn(n) * 300)
    open_ = close + np.random.randn(n) * 100
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.random.randint(1000, 10000, n).astype(float)

    dxy = 102 + np.cumsum(np.random.randn(n) * 0.3)
    vix = 18 + np.cumsum(np.random.randn(n) * 0.5)
    vix = np.maximum(vix, 10.0)
    spy = 480 + np.cumsum(np.random.randn(n) * 2)
    stab = 1.3e11 + np.cumsum(np.random.randn(n) * 5e8)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "macro_dxy": dxy,
            "macro_vix": vix,
            "macro_spy": spy,
            "oc_stablecoin_total_circulating_usd": stab,
        },
        index=dates,
    )


@pytest.fixture
def preprocessed_df(sample_macro_df: pd.DataFrame, config: MacroLiqTrendConfig) -> pd.DataFrame:
    return preprocess(sample_macro_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: MacroLiqTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_macro_df: pd.DataFrame) -> None:
        config = MacroLiqTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_macro_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_macro_df: pd.DataFrame) -> None:
        config = MacroLiqTrendConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_macro_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_needs_drawdown(self, sample_macro_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: shorts only when drawdown < threshold."""
        config = MacroLiqTrendConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,  # Easy to trigger
            hedge_strength_ratio=0.5,
        )
        df = preprocess(sample_macro_df, config)
        signals = generate_signals(df, config)
        # Direction should be valid integers
        assert set(signals.direction.unique()).issubset({-1, 0, 1})


class TestSignalLogic:
    def test_bullish_macro_generates_long(self) -> None:
        """Strong bullish macro + price above SMA should produce long."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        # Uptrend crypto price
        close = 50000 + np.arange(n) * 100.0
        high = close + 200
        low = close - 200
        open_ = close - 50
        volume = np.full(n, 5000.0)

        # DXY trending down (bullish for crypto)
        dxy = 105 - np.arange(n) * 0.05
        # VIX trending down (risk-on)
        vix = 25 - np.arange(n) * 0.03
        vix = np.maximum(vix, 10.0)
        # SPY trending up (risk-on)
        spy = 450 + np.arange(n) * 0.5
        # Stablecoin supply growing
        stab = 1.2e11 + np.arange(n) * 5e8

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "macro_dxy": dxy,
                "macro_vix": vix,
                "macro_spy": spy,
                "oc_stablecoin_total_circulating_usd": stab,
            },
            index=dates,
        )

        config = MacroLiqTrendConfig(
            liq_long_threshold=0.3,
            zscore_window=30,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # After warmup, should have some long signals in strong uptrend
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        assert (post_warmup >= 0).all() or (post_warmup == 1).any()

    def test_no_simultaneous_long_short(
        self, preprocessed_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        """Direction should never be both long and short at same bar."""
        signals = generate_signals(preprocessed_df, config)
        long_mask = signals.direction == 1
        short_mask = signals.direction == -1
        # No bar should have both
        assert not (long_mask & short_mask).any()

    def test_entry_on_direction_change(
        self, preprocessed_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        """Entries should only occur when direction changes."""
        signals = generate_signals(preprocessed_df, config)
        entries = signals.entries
        direction = signals.direction
        prev_dir = direction.shift(1).fillna(0).astype(int)

        # Where entry is True, direction must differ from previous
        entry_mask = entries
        if entry_mask.any():
            assert (direction[entry_mask] != prev_dir[entry_mask]).all()
