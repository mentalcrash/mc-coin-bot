"""Tests for Momentum Acceleration signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.mom_accel.config import MomAccelConfig, ShortMode
from src.strategy.mom_accel.preprocessor import preprocess
from src.strategy.mom_accel.signal import generate_signals


def _make_df() -> pd.DataFrame:
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
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestSignalStructure:
    def test_output_fields(self) -> None:
        config = MomAccelConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self) -> None:
        config = MomAccelConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self) -> None:
        config = MomAccelConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self) -> None:
        config = MomAccelConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        n = len(df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(self) -> None:
        config = MomAccelConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self) -> None:
        config = MomAccelConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self) -> None:
        config = MomAccelConfig(short_mode=ShortMode.FULL)
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int
