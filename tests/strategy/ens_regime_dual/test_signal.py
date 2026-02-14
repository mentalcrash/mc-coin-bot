"""Tests for Regime-Adaptive Dual-Alpha Ensemble signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.ens_regime_dual.strategy import EnsRegimeDualStrategy


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 500
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


class TestSignalStructure:
    def test_output_fields(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = EnsRegimeDualStrategy()
        _processed, signals = strategy.run(sample_ohlcv_df)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = EnsRegimeDualStrategy()
        _, signals = strategy.run(sample_ohlcv_df)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = EnsRegimeDualStrategy()
        _, signals = strategy.run(sample_ohlcv_df)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = EnsRegimeDualStrategy()
        _, signals = strategy.run(sample_ohlcv_df)
        n = len(sample_ohlcv_df)
        assert len(signals.entries) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n
