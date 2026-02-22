"""Tests for Stablecoin Velocity Regime config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.stablecoin_velocity.config import ShortMode, StablecoinVelocityConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestStablecoinVelocityConfig:
    def test_default_values(self) -> None:
        config = StablecoinVelocityConfig()
        assert config.velocity_fast_window == 7
        assert config.velocity_slow_window == 30
        assert config.zscore_window == 60
        assert config.zscore_entry_threshold == 1.0
        assert config.zscore_exit_threshold == -1.0
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = StablecoinVelocityConfig()
        with pytest.raises(ValidationError):
            config.velocity_fast_window = 999  # type: ignore[misc]

    def test_velocity_fast_window_range(self) -> None:
        with pytest.raises(ValidationError):
            StablecoinVelocityConfig(velocity_fast_window=1)
        with pytest.raises(ValidationError):
            StablecoinVelocityConfig(velocity_fast_window=31)

    def test_velocity_slow_window_range(self) -> None:
        with pytest.raises(ValidationError):
            StablecoinVelocityConfig(velocity_slow_window=5)
        with pytest.raises(ValidationError):
            StablecoinVelocityConfig(velocity_slow_window=121)

    def test_fast_must_be_less_than_slow(self) -> None:
        with pytest.raises(ValidationError):
            StablecoinVelocityConfig(velocity_fast_window=30, velocity_slow_window=30)

    def test_zscore_entry_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            StablecoinVelocityConfig(zscore_entry_threshold=0.2)
        with pytest.raises(ValidationError):
            StablecoinVelocityConfig(zscore_entry_threshold=3.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            StablecoinVelocityConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = StablecoinVelocityConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.zscore_window

    def test_annualization_factor(self) -> None:
        config = StablecoinVelocityConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = StablecoinVelocityConfig(velocity_fast_window=5, velocity_slow_window=20)
        assert config.velocity_fast_window == 5
        assert config.velocity_slow_window == 20
