"""Tests for LarryVBConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.larry_vb.config import LarryVBConfig, ShortMode


class TestConfigDefaults:
    """Default value tests."""

    def test_default_values(self) -> None:
        """Default config has expected values."""
        config = LarryVBConfig()

        assert config.k_factor == 0.5
        assert config.vol_window == 20
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen_model(self) -> None:
        """Config is immutable (frozen=True)."""
        config = LarryVBConfig()

        with pytest.raises(ValidationError):
            config.k_factor = 0.7  # type: ignore[misc]


class TestConfigValidation:
    """Validation tests."""

    def test_vol_target_below_min_volatility_raises(self) -> None:
        """vol_target must be >= min_volatility."""
        with pytest.raises(ValidationError, match="vol_target"):
            LarryVBConfig(vol_target=0.03, min_volatility=0.05)

    def test_vol_target_below_absolute_minimum_raises(self) -> None:
        """vol_target below absolute min (0.05) raises."""
        with pytest.raises(ValidationError):
            LarryVBConfig(vol_target=0.01)

    def test_k_factor_below_minimum_raises(self) -> None:
        """k_factor below 0.1 raises."""
        with pytest.raises(ValidationError):
            LarryVBConfig(k_factor=0.05)

    def test_k_factor_above_maximum_raises(self) -> None:
        """k_factor above 1.0 raises."""
        with pytest.raises(ValidationError):
            LarryVBConfig(k_factor=1.5)

    def test_vol_window_below_minimum_raises(self) -> None:
        """vol_window below 5 raises."""
        with pytest.raises(ValidationError):
            LarryVBConfig(vol_window=3)

    def test_vol_window_above_maximum_raises(self) -> None:
        """vol_window above 100 raises."""
        with pytest.raises(ValidationError):
            LarryVBConfig(vol_window=200)

    def test_annualization_factor_zero_raises(self) -> None:
        """annualization_factor must be > 0."""
        with pytest.raises(ValidationError):
            LarryVBConfig(annualization_factor=0)

    def test_valid_custom_config(self) -> None:
        """Valid custom config is created successfully."""
        config = LarryVBConfig(
            k_factor=0.7,
            vol_window=30,
            vol_target=0.50,
            short_mode=ShortMode.DISABLED,
        )
        assert config.k_factor == 0.7
        assert config.vol_window == 30
        assert config.vol_target == 0.50
        assert config.short_mode == ShortMode.DISABLED


class TestConfigWarmup:
    """Warmup period tests."""

    def test_warmup_periods_default(self) -> None:
        """warmup_periods = vol_window + 2."""
        config = LarryVBConfig()
        assert config.warmup_periods() == 22  # 20 + 2

    def test_warmup_periods_custom(self) -> None:
        """Custom vol_window changes warmup."""
        config = LarryVBConfig(vol_window=50)
        assert config.warmup_periods() == 52  # 50 + 2


class TestConfigTimeframe:
    """for_timeframe factory tests."""

    def test_for_timeframe_1d(self) -> None:
        """for_timeframe('1d') sets annualization_factor=365.0."""
        config = LarryVBConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_4h(self) -> None:
        """for_timeframe('4h') sets annualization_factor=2190.0."""
        config = LarryVBConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_1h(self) -> None:
        """for_timeframe('1h') sets annualization_factor=8760.0."""
        config = LarryVBConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe accepts extra kwargs."""
        config = LarryVBConfig.for_timeframe("1d", k_factor=0.7, vol_target=0.30)
        assert config.k_factor == 0.7
        assert config.vol_target == 0.30
        assert config.annualization_factor == 365.0

    def test_for_timeframe_unknown(self) -> None:
        """Unknown timeframe defaults to 365.0."""
        config = LarryVBConfig.for_timeframe("2h")
        assert config.annualization_factor == 365.0


class TestConfigPresets:
    """Preset factory method tests."""

    def test_conservative_preset(self) -> None:
        """conservative() returns high k_factor, low vol_target config."""
        config = LarryVBConfig.conservative()

        assert config.k_factor == 0.7
        assert config.vol_target == 0.30

    def test_aggressive_preset(self) -> None:
        """aggressive() returns low k_factor, high vol_target config."""
        config = LarryVBConfig.aggressive()

        assert config.k_factor == 0.3
        assert config.vol_target == 0.50
