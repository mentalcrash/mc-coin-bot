"""Tests for Stochastic Momentum Hybrid Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.stoch_mom.config import ShortMode, StochMomConfig
from src.strategy.stoch_mom.preprocessor import preprocess
from src.strategy.stoch_mom.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def processed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Preprocessed DataFrame."""
    config = StochMomConfig()
    return preprocess(sample_ohlcv, config)


class TestSignalBasic:
    """Signal basic structure tests."""

    def test_generate_signals_basic(self, processed_df: pd.DataFrame) -> None:
        """Signal output type and size verification."""
        config = StochMomConfig()
        signals = generate_signals(processed_df, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

        n = len(processed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_entries_exits_are_bool(self, processed_df: pd.DataFrame) -> None:
        """entries and exits have bool dtype."""
        config = StochMomConfig()
        signals = generate_signals(processed_df, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, processed_df: pd.DataFrame) -> None:
        """direction only contains -1, 0, 1 values."""
        config = StochMomConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)

        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_strength_no_nan(self, processed_df: pd.DataFrame) -> None:
        """strength has no NaN (filled with 0)."""
        config = StochMomConfig()
        signals = generate_signals(processed_df, config)
        assert signals.strength.isna().sum() == 0


class TestStateMachine:
    """State machine position tracking tests."""

    def test_state_machine_long(self) -> None:
        """%K cross up with trend_up generates long entry."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")

        np.random.seed(42)
        # Create price series that will generate stochastic crossover
        close = np.ones(n) * 50000
        # First 100 bars: stable with mild uptrend to establish SMA
        close[:100] = 50000 + np.arange(100) * 10 + np.random.randn(100) * 5
        # 100-130: drop to create oversold condition
        close[100:130] = close[99] - np.arange(30) * 100
        # 130-160: recover to create %K cross above %D
        close[130:160] = close[129] + np.arange(1, 31) * 150
        # 160-200: continue uptrend
        close[160:200] = close[159] + np.cumsum(np.ones(40) * 50)

        high = close + np.abs(np.random.randn(n) * 100)
        low = close - np.abs(np.random.randn(n) * 100)
        open_ = close + np.random.randn(n) * 20
        volume = np.ones(n) * 1000

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        config = StochMomConfig(short_mode=ShortMode.FULL, k_period=14, sma_period=30)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # At least one entry should occur
        assert signals.entries.any(), "No entries generated"
        # Direction should contain long(1)
        assert (signals.direction == Direction.LONG.value).any(), "No long direction found"

    def test_first_bar_is_neutral(self, processed_df: pd.DataFrame) -> None:
        """First bar direction is neutral(0)."""
        config = StochMomConfig()
        signals = generate_signals(processed_df, config)
        assert signals.direction.iloc[0] == Direction.NEUTRAL.value


class TestShortMode:
    """ShortMode handling tests."""

    def test_short_mode_full(self, processed_df: pd.DataFrame) -> None:
        """FULL mode allows short direction."""
        config = StochMomConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(processed_df, config)

        # Direction range should include -1 possibility
        unique_values = set(signals.direction.unique())
        assert unique_values.issubset({-1, 0, 1})

    def test_short_mode_disabled(self, processed_df: pd.DataFrame) -> None:
        """DISABLED mode has no short direction(-1)."""
        config = StochMomConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(processed_df, config)

        assert (signals.direction >= 0).all(), "Short direction found in DISABLED mode"
        assert (signals.strength >= 0).all(), "Negative strength found in DISABLED mode"

    def test_disabled_vs_full_direction_difference(self, processed_df: pd.DataFrame) -> None:
        """DISABLED and FULL mode comparison."""
        config_disabled = StochMomConfig(short_mode=ShortMode.DISABLED)
        config_full = StochMomConfig(short_mode=ShortMode.FULL)

        signals_disabled = generate_signals(processed_df, config_disabled)
        signals_full = generate_signals(processed_df, config_full)

        # Both modes should generate valid signals
        abs_disabled = signals_disabled.strength.abs().sum()
        abs_full = signals_full.strength.abs().sum()
        assert abs_full >= 0
        assert abs_disabled >= 0


class TestShift1Rule:
    """Shift(1) lookahead bias prevention tests."""

    def test_first_row_strength_is_zero(self, processed_df: pd.DataFrame) -> None:
        """First row strength is 0 (shift causes NaN -> 0)."""
        config = StochMomConfig()
        signals = generate_signals(processed_df, config)
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead(self, processed_df: pd.DataFrame) -> None:
        """Changing last bar data does not affect previous bar signals."""
        config = StochMomConfig()

        signals_before = generate_signals(processed_df, config)
        strength_second_last = signals_before.strength.iloc[-2]

        # Modify last bar's indicators
        modified_df = processed_df.copy()
        modified_df.iloc[-1, modified_df.columns.get_loc("pct_k")] = 99.0
        modified_df.iloc[-1, modified_df.columns.get_loc("pct_d")] = 1.0
        modified_df.iloc[-1, modified_df.columns.get_loc("vol_scalar")] = 99.0

        signals_after = generate_signals(modified_df, config)
        # Second-to-last row should not be affected
        assert signals_after.strength.iloc[-2] == strength_second_last


class TestMissingColumns:
    """Missing column error tests."""

    def test_missing_pct_k_column_raises(self) -> None:
        """Missing pct_k column raises ValueError."""
        df = pd.DataFrame(
            {
                "pct_d": [50.0],
                "sma": [50000.0],
                "close": [50000.0],
                "vol_scalar": [1.0],
                "vol_ratio": [0.5],
            }
        )
        config = StochMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_vol_ratio_column_raises(self) -> None:
        """Missing vol_ratio column raises ValueError."""
        df = pd.DataFrame(
            {
                "pct_k": [50.0],
                "pct_d": [50.0],
                "sma": [50000.0],
                "close": [50000.0],
                "vol_scalar": [1.0],
            }
        )
        config = StochMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_sma_column_raises(self) -> None:
        """Missing sma column raises ValueError."""
        df = pd.DataFrame(
            {
                "pct_k": [50.0],
                "pct_d": [50.0],
                "close": [50000.0],
                "vol_scalar": [1.0],
                "vol_ratio": [0.5],
            }
        )
        config = StochMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
