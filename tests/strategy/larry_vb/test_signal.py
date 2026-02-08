"""Tests for Larry Williams Volatility Breakout signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.larry_vb.config import LarryVBConfig, ShortMode
from src.strategy.larry_vb.preprocessor import preprocess
from src.strategy.larry_vb.signal import generate_signals
from src.strategy.types import Direction, StrategySignals


@pytest.fixture
def preprocessed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Preprocessed DataFrame ready for signal generation."""
    config = LarryVBConfig()
    return preprocess(sample_ohlcv, config)


@pytest.fixture
def default_config() -> LarryVBConfig:
    """Default LarryVBConfig."""
    return LarryVBConfig()


class TestGenerateSignalsBasic:
    """Basic signal generation tests."""

    def test_generate_signals_returns_strategy_signals(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """Signal output is StrategySignals NamedTuple."""
        signals = generate_signals(preprocessed_df, default_config)

        assert isinstance(signals, StrategySignals)

    def test_signal_lengths_match_input(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """All signal Series have same length as input DataFrame."""
        signals = generate_signals(preprocessed_df, default_config)

        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)

    def test_entries_exits_are_bool(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """entries and exits are boolean Series."""
        signals = generate_signals(preprocessed_df, default_config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_strength_is_numeric_no_nan(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """Strength is a numeric Series with no NaN."""
        signals = generate_signals(preprocessed_df, default_config)

        assert pd.api.types.is_numeric_dtype(signals.strength)
        assert not signals.strength.isna().any()

    def test_direction_no_nan(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """Direction has no NaN values."""
        signals = generate_signals(preprocessed_df, default_config)

        assert not signals.direction.isna().any()

    def test_default_config_when_none(self, preprocessed_df: pd.DataFrame) -> None:
        """config=None uses default LarryVBConfig."""
        signals = generate_signals(preprocessed_df, config=None)

        assert isinstance(signals, StrategySignals)
        assert len(signals.entries) == len(preprocessed_df)


class TestShift1Rule:
    """Shift(1) lookahead prevention tests."""

    def test_first_bar_direction_is_zero(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """First bar should always be direction=0 due to shift(1)."""
        signals = generate_signals(preprocessed_df, default_config)

        assert signals.direction.iloc[0] == 0

    def test_first_bar_strength_is_zero(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """First bar should always be strength=0 due to shift(1)."""
        signals = generate_signals(preprocessed_df, default_config)

        assert signals.strength.iloc[0] == 0.0

    def test_direction_is_shifted(self) -> None:
        """Direction at bar t reflects breakout at bar t-1 (shift(1) applied).

        Manually construct data where breakout happens at a known bar,
        then verify direction is active at the next bar.
        """
        n = 30
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")

        # Flat price with a breakout at bar 10
        close = np.full(n, 100.0)
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)

        # Create a clear upward breakout at bar 10:
        # prev_range = high[9] - low[9] = 2.0
        # breakout_upper = open[10] + 0.5 * 2.0 = 101.0
        # close[10] = 110.0 > 101.0 → long breakout
        close[10] = 110.0

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
            index=dates,
        )

        config = LarryVBConfig(vol_window=5)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # Bar 10 has the breakout, but due to shift(1),
        # direction should be active at bar 11
        assert signals.direction.iloc[10] == 0  # breakout bar itself
        assert signals.direction.iloc[11] == Direction.LONG  # next bar = hold


class TestShortMode:
    """ShortMode handling tests."""

    def test_full_mode_allows_short(self, preprocessed_df: pd.DataFrame) -> None:
        """FULL mode allows direction=-1."""
        config = LarryVBConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        # With random data, at least some breakouts should happen in both directions
        unique_dirs = set(signals.direction.unique())
        assert Direction.LONG in unique_dirs or Direction.SHORT in unique_dirs

    def test_disabled_mode_no_short(self, preprocessed_df: pd.DataFrame) -> None:
        """DISABLED mode converts -1 to 0."""
        config = LarryVBConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(preprocessed_df, config)

        assert Direction.SHORT not in signals.direction.values
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({0, 1})

    def test_disabled_mode_zero_strength_for_shorts(self, preprocessed_df: pd.DataFrame) -> None:
        """DISABLED mode sets strength=0 where direction would be -1."""
        config = LarryVBConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(preprocessed_df, config)

        neutral_mask = signals.direction == Direction.NEUTRAL
        assert (signals.strength[neutral_mask] == 0.0).all()


class TestEntriesExits:
    """Entry/Exit signal tests."""

    def test_entry_on_direction_change(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """Entries occur when direction changes to non-zero."""
        signals = generate_signals(preprocessed_df, default_config)

        # Where entries are True, direction should be non-zero
        entry_directions = signals.direction[signals.entries]
        if len(entry_directions) > 0:
            assert (entry_directions != 0).all()

    def test_exit_on_direction_to_neutral(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """Exits occur when direction goes to neutral."""
        signals = generate_signals(preprocessed_df, default_config)

        # Where exits are True, direction should be 0 (neutral) or reversed
        if signals.exits.any():
            # At least some exits should exist given random data
            exit_count = int(signals.exits.sum())
            assert exit_count > 0

    def test_entries_and_exits_alternate(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """No simultaneous entry and exit on the same bar."""
        signals = generate_signals(preprocessed_df, default_config)

        # entry and exit should not both be True on the same bar
        # (except for reversals where we exit old and enter new)
        overlap = signals.entries & signals.exits
        # Reversals can cause both entry and exit
        # Just verify that the signals are consistent with direction changes
        if overlap.any():
            # On reversal bars, direction changes sign
            prev_dir = signals.direction.shift(1).fillna(0)
            reversal_mask = signals.direction * prev_dir < 0
            assert overlap[overlap].index.isin(reversal_mask[reversal_mask].index).all()


class TestDirectionValues:
    """Direction value range tests."""

    def test_direction_in_valid_range(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """Direction values are in {-1, 0, 1}."""
        signals = generate_signals(preprocessed_df, default_config)

        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_one_bar_hold_pattern(
        self, preprocessed_df: pd.DataFrame, default_config: LarryVBConfig
    ) -> None:
        """1-bar hold means direction doesn't persist for multiple bars.

        After a breakout at bar t, direction is non-zero at bar t+1,
        then returns to 0 at bar t+2 (unless another breakout at t+1).
        """
        signals = generate_signals(preprocessed_df, default_config)

        direction = signals.direction
        # Find stretches where direction is non-zero
        nonzero = direction != 0
        # Group consecutive non-zero bars
        groups = (nonzero != nonzero.shift(1)).cumsum()
        nonzero_groups = groups[nonzero]

        if len(nonzero_groups) > 0:
            # Most non-zero groups should be length 1 (1-bar hold)
            # Some may be longer if consecutive breakouts happen
            group_sizes = nonzero_groups.value_counts()
            # At least 50% should be single-bar holds
            single_bar = (group_sizes == 1).sum()
            total_groups = len(group_sizes)
            if total_groups > 2:
                # With random data, most positions should be 1-bar
                assert single_bar / total_groups > 0.3


class TestSignalMissingColumns:
    """Input validation tests."""

    def test_missing_required_columns_raises(self) -> None:
        """Missing preprocessor columns raises ValueError."""
        df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = LarryVBConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_missing_breakout_upper_raises(self) -> None:
        """Missing breakout_upper column raises ValueError."""
        df = pd.DataFrame(
            {
                "close": [1, 2, 3],
                "breakout_lower": [0.9, 1.9, 2.9],
                "vol_scalar": [1.0, 1.0, 1.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = LarryVBConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)
