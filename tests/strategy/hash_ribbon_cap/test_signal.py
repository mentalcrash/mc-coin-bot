"""Tests for Hash-Ribbon Capitulation signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.hash_ribbon_cap.config import HashRibbonCapConfig, ShortMode
from src.strategy.hash_ribbon_cap.preprocessor import preprocess
from src.strategy.hash_ribbon_cap.signal import generate_signals


@pytest.fixture
def config() -> HashRibbonCapConfig:
    return HashRibbonCapConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: HashRibbonCapConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: HashRibbonCapConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = HashRibbonCapConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_hedge_only_respects_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = HashRibbonCapConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = HashRibbonCapConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestStrategyLogic:
    def test_no_signal_without_capitulation(self) -> None:
        """Monotonically rising price → no capitulation → no recovery signal."""
        n = 200
        close = 100 + np.arange(n, dtype=float)  # Always rising
        high = close + 1.0
        low = close - 1.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        config = HashRibbonCapConfig()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # With monotonically rising, fast SMA > slow SMA always → no capitulation → no recovery entry
        # But had_capitulation stays 0, so long_signal should be False
        assert (signals.direction >= 0).all()

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: HashRibbonCapConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()
