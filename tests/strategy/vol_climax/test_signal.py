"""Tests for Volume Climax Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.types import Direction
from src.strategy.vol_climax.config import ShortMode, VolClimaxConfig
from src.strategy.vol_climax.preprocessor import preprocess
from src.strategy.vol_climax.signal import generate_signals


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    # Insert volume spikes to trigger climax
    volume[50] = 50000.0
    volume[100] = 60000.0
    volume[200] = 45000.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestSignalStructure:
    def test_output_structure(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_ohlcv_df)


class TestShift1Rule:
    def test_first_bar_no_signal(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert signals.direction.dtype == int

    def test_hedge_only_suppresses_without_drawdown(self):
        """Hedge-only mode: no shorts when no drawdown."""
        n = 200
        # Monotonically increasing price = no drawdown
        close = np.linspace(100, 200, n)
        high = close + 2.0
        low = close - 2.0

        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        config = VolClimaxConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = VolClimaxConfig(short_mode=ShortMode.FULL)
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed)
        assert len(signals.entries) == len(sample_ohlcv_df)
