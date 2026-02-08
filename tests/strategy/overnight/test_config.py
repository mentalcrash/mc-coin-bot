"""Tests for OvernightConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.overnight.config import OvernightConfig
from src.strategy.tsmom.config import ShortMode


class TestOvernightConfigDefaults:
    """Default value tests."""

    def test_default_values(self) -> None:
        """Default config has expected values."""
        config = OvernightConfig()

        assert config.entry_hour == 22
        assert config.exit_hour == 0
        assert config.vol_target == 0.30
        assert config.annualization_factor == 8760.0
        assert config.use_vol_filter is False
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen_model(self) -> None:
        """Config is immutable (frozen=True)."""
        config = OvernightConfig()

        with pytest.raises(ValidationError):
            config.entry_hour = 10  # type: ignore[misc]


class TestOvernightConfigValidation:
    """Validation tests."""

    def test_entry_exit_same_hour_raises(self) -> None:
        """entry_hour == exit_hour raises ValidationError."""
        with pytest.raises(ValidationError, match="must differ"):
            OvernightConfig(entry_hour=10, exit_hour=10)

    def test_vol_target_validation(self) -> None:
        """vol_target must be >= min_volatility."""
        with pytest.raises(ValidationError, match="vol_target"):
            OvernightConfig(vol_target=0.03, min_volatility=0.05)

    def test_vol_target_below_minimum_raises(self) -> None:
        """vol_target below absolute min (0.05) raises."""
        with pytest.raises(ValidationError):
            OvernightConfig(vol_target=0.01)

    def test_entry_hour_out_of_range_raises(self) -> None:
        """entry_hour outside 0-23 raises."""
        with pytest.raises(ValidationError):
            OvernightConfig(entry_hour=25)

    def test_exit_hour_out_of_range_raises(self) -> None:
        """exit_hour outside 0-23 raises."""
        with pytest.raises(ValidationError):
            OvernightConfig(exit_hour=-1)


class TestOvernightConfigMethods:
    """Method tests."""

    def test_warmup_periods(self) -> None:
        """warmup_periods = vol_window + 1."""
        config = OvernightConfig(vol_window=30)
        assert config.warmup_periods() == 31

        config2 = OvernightConfig(vol_window=48)
        assert config2.warmup_periods() == 49

    def test_for_timeframe_1h(self) -> None:
        """for_timeframe('1h') sets annualization_factor=8760.0."""
        config = OvernightConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_4h(self) -> None:
        """for_timeframe('4h') sets annualization_factor=2190.0."""
        config = OvernightConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe accepts extra kwargs."""
        config = OvernightConfig.for_timeframe("1h", vol_target=0.20, entry_hour=20)
        assert config.vol_target == 0.20
        assert config.entry_hour == 20
        assert config.annualization_factor == 8760.0


class TestOvernightConfigPresets:
    """Preset factory method tests."""

    def test_conservative_preset(self) -> None:
        """conservative() returns low vol_target config."""
        config = OvernightConfig.conservative()

        assert config.vol_target == 0.15
        assert config.min_volatility == 0.08
        assert config.vol_window == 48
        assert config.use_vol_filter is False

    def test_aggressive_preset(self) -> None:
        """aggressive() returns high vol_target config with vol filter."""
        config = OvernightConfig.aggressive()

        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.vol_window == 20
        assert config.use_vol_filter is True
        assert config.vol_filter_threshold == 1.3
