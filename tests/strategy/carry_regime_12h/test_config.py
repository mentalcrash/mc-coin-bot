"""Tests for Carry-Regime Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.carry_regime_12h.config import CarryRegimeConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCarryRegimeConfig:
    def test_default_values(self) -> None:
        config = CarryRegimeConfig()
        assert config.ema_fast == 8
        assert config.ema_mid == 21
        assert config.ema_slow == 55
        assert config.fr_percentile_window == 135
        assert config.carry_sensitivity == 0.5
        assert config.exit_base_threshold == 0.3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = CarryRegimeConfig()
        with pytest.raises(ValidationError):
            config.ema_fast = 999  # type: ignore[misc]

    def test_ema_fast_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryRegimeConfig(ema_fast=2)  # below ge=3
        with pytest.raises(ValidationError):
            CarryRegimeConfig(ema_fast=31, ema_mid=21)  # above le=30

    def test_ema_mid_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryRegimeConfig(ema_mid=9)  # below ge=10
        with pytest.raises(ValidationError):
            CarryRegimeConfig(ema_mid=61)  # above le=60

    def test_ema_slow_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryRegimeConfig(ema_slow=29)  # below ge=30
        with pytest.raises(ValidationError):
            CarryRegimeConfig(ema_slow=151)  # above le=150

    def test_ema_ordering_fast_ge_mid(self) -> None:
        with pytest.raises(ValidationError, match="ema_fast"):
            CarryRegimeConfig(ema_fast=21, ema_mid=21)

    def test_ema_ordering_mid_ge_slow(self) -> None:
        with pytest.raises(ValidationError, match="ema_mid"):
            CarryRegimeConfig(ema_mid=55, ema_slow=55)

    def test_fr_percentile_window_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryRegimeConfig(fr_percentile_window=29)
        with pytest.raises(ValidationError):
            CarryRegimeConfig(fr_percentile_window=401)

    def test_carry_sensitivity_range(self) -> None:
        config_zero = CarryRegimeConfig(carry_sensitivity=0.0)
        assert config_zero.carry_sensitivity == 0.0
        with pytest.raises(ValidationError):
            CarryRegimeConfig(carry_sensitivity=-0.1)
        with pytest.raises(ValidationError):
            CarryRegimeConfig(carry_sensitivity=2.1)

    def test_exit_base_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            CarryRegimeConfig(exit_base_threshold=-0.1)
        with pytest.raises(ValidationError):
            CarryRegimeConfig(exit_base_threshold=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError, match="vol_target"):
            CarryRegimeConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = CarryRegimeConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.ema_slow
        assert config.warmup_periods() >= config.fr_percentile_window

    def test_annualization_factor(self) -> None:
        config = CarryRegimeConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = CarryRegimeConfig(ema_fast=5, ema_mid=15, ema_slow=40)
        assert config.ema_fast == 5
        assert config.ema_mid == 15
        assert config.ema_slow == 40

    def test_pure_trend_mode(self) -> None:
        """carry_sensitivity=0 → pure trend baseline."""
        config = CarryRegimeConfig(carry_sensitivity=0.0)
        assert config.carry_sensitivity == 0.0

    def test_hedge_params(self) -> None:
        config = CarryRegimeConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5
