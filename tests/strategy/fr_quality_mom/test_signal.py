"""Tests for FR Quality Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.fr_quality_mom.config import FrQualityMomConfig, ShortMode
from src.strategy.fr_quality_mom.preprocessor import preprocess
from src.strategy.fr_quality_mom.signal import generate_signals


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
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestSignalStructure:
    def test_output_fields(self) -> None:
        config = FrQualityMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self) -> None:
        config = FrQualityMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self) -> None:
        config = FrQualityMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self) -> None:
        config = FrQualityMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        n = len(df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(self) -> None:
        config = FrQualityMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self) -> None:
        config = FrQualityMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self) -> None:
        config = FrQualityMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestQualityFilter:
    def test_high_crowding_reduces_signals(self) -> None:
        """FR crowding이 높으면 시그널이 차단되어야 한다."""
        # Low threshold = more filtering
        config_strict = FrQualityMomConfig(fr_crowd_threshold=0.5)
        # High threshold = less filtering
        config_lenient = FrQualityMomConfig(fr_crowd_threshold=3.0)
        df_raw = _make_df()
        df_strict = preprocess(df_raw, config_strict)
        df_lenient = preprocess(df_raw, config_lenient)
        sig_strict = generate_signals(df_strict, config_strict)
        sig_lenient = generate_signals(df_lenient, config_lenient)
        # strict should have fewer or equal non-zero directions
        active_strict = (sig_strict.direction != 0).sum()
        active_lenient = (sig_lenient.direction != 0).sum()
        assert active_strict <= active_lenient
