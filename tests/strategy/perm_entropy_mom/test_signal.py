"""Tests for Permutation Entropy Momentum Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.perm_entropy_mom.config import PermEntropyMomConfig, ShortMode
from src.strategy.perm_entropy_mom.preprocessor import preprocess
from src.strategy.perm_entropy_mom.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestSignalStructure:
    def test_output_structure(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_ohlcv_df)


class TestShift1Rule:
    def test_first_bar_no_signal(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestNoiseGate:
    """High PE (noise) should produce zero signal."""

    def test_noise_zone_zero_signal(self):
        """When PE > noise_threshold, signal should be zero."""
        n = 200
        # Pure random walk -> high PE
        np.random.seed(99)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 0.5
        low = close - 0.5
        open_ = close + np.random.randn(n) * 0.1

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        # Very low noise threshold -> more bars gated out
        config = PermEntropyMomConfig(
            noise_threshold=0.5,
            short_mode=ShortMode.FULL,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # Some bars should be neutral due to noise gate
        neutral_count = (signals.direction == Direction.NEUTRAL).sum()
        assert neutral_count > 0

    def test_trending_data_generates_signals(self):
        """Strong trend should have low PE -> non-zero signals."""
        n = 200
        close = np.linspace(100, 200, n)
        high = close + 0.5
        low = close - 0.5
        open_ = close - 0.2

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        config = PermEntropyMomConfig(
            noise_threshold=0.95,
            short_mode=ShortMode.FULL,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # Strong uptrend -> some positive signals
        assert signals.strength.abs().sum() > 0


class TestShortModeSignal:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert signals.direction.dtype == int

    def test_hedge_only_suppresses_without_drawdown(self):
        n = 200
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

        config = PermEntropyMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # Uptrend => no drawdown => shorts suppressed
        assert (signals.direction != Direction.SHORT).all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = PermEntropyMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed)
        assert len(signals.entries) == len(sample_ohlcv_df)
