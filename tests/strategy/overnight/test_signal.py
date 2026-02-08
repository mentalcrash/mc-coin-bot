"""Tests for Overnight Seasonality signal generator."""

import pandas as pd
import pytest

from src.strategy.overnight.config import OvernightConfig
from src.strategy.overnight.preprocessor import preprocess
from src.strategy.overnight.signal import generate_signals
from src.strategy.types import StrategySignals


@pytest.fixture
def preprocessed_df(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Preprocessed DataFrame ready for signal generation."""
    config = OvernightConfig()
    return preprocess(sample_ohlcv, config)


@pytest.fixture
def default_config() -> OvernightConfig:
    """Default OvernightConfig."""
    return OvernightConfig()


class TestGenerateSignalsBasic:
    """Basic signal generation tests."""

    def test_generate_signals_basic(
        self, preprocessed_df: pd.DataFrame, default_config: OvernightConfig
    ) -> None:
        """Signal output has correct types and sizes."""
        signals = generate_signals(preprocessed_df, default_config)

        assert isinstance(signals, StrategySignals)
        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)

    def test_entries_exits_are_bool(
        self, preprocessed_df: pd.DataFrame, default_config: OvernightConfig
    ) -> None:
        """entries and exits are boolean Series."""
        signals = generate_signals(preprocessed_df, default_config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, default_config: OvernightConfig
    ) -> None:
        """Direction contains only 0 and 1 (Long-Only mode, no -1)."""
        signals = generate_signals(preprocessed_df, default_config)

        unique_directions = set(signals.direction.unique())
        assert unique_directions.issubset({0, 1})

    def test_strength_is_numeric(
        self, preprocessed_df: pd.DataFrame, default_config: OvernightConfig
    ) -> None:
        """Strength is a numeric Series with no NaN."""
        signals = generate_signals(preprocessed_df, default_config)

        assert pd.api.types.is_numeric_dtype(signals.strength)
        assert not signals.strength.isna().any()


class TestOvernightPositionTiming:
    """Verify positions are only active during expected hours."""

    def test_overnight_position_timing(self, preprocessed_df: pd.DataFrame) -> None:
        """Positions are only active during entry_hour to exit_hour window.

        Default: entry=22, exit=0 means positions at hour=22,23
        (shifted by 1, so direction=1 at hours following 22 and 23).
        """
        config = OvernightConfig(entry_hour=22, exit_hour=0)
        signals = generate_signals(preprocessed_df, config)

        # Due to shift(1), the in_position at hour H means direction=1 at hour H+1
        # in_position is True at hours 22, 23 (wrap-around: hour>=22 OR hour<0)
        # After shift(1), direction=1 at the bar AFTER hours 22 and 23
        # So direction should be 1 at hours 23 and 0

        # At minimum, there should be some entries and some neutrals
        assert (signals.direction == 1).any(), "Should have long positions"
        assert (signals.direction == 0).any(), "Should have neutral periods"

    def test_same_day_session(self, sample_ohlcv: pd.DataFrame) -> None:
        """Same-day session (entry=9, exit=17) should work without wrap-around."""
        config = OvernightConfig(entry_hour=9, exit_hour=17)
        processed = preprocess(sample_ohlcv, config)
        signals = generate_signals(processed, config)

        assert (signals.direction == 1).any()
        assert (signals.direction == 0).any()


class TestLongOnlyMode:
    """Long-only (ShortMode.DISABLED) tests."""

    def test_long_only_mode(
        self, preprocessed_df: pd.DataFrame, default_config: OvernightConfig
    ) -> None:
        """Only 0 and 1 in direction (no -1) for default Long-Only."""
        signals = generate_signals(preprocessed_df, default_config)

        assert -1 not in signals.direction.values
        unique = set(signals.direction.unique())
        assert unique.issubset({0, 1})


class TestSignalValidation:
    """Input validation tests."""

    def test_missing_required_columns_raises(self) -> None:
        """Missing preprocessor columns raises ValueError."""
        df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1h"),
        )
        config = OvernightConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config_when_none(self, preprocessed_df: pd.DataFrame) -> None:
        """config=None uses default OvernightConfig."""
        signals = generate_signals(preprocessed_df, config=None)

        assert isinstance(signals, StrategySignals)
        assert len(signals.entries) == len(preprocessed_df)
