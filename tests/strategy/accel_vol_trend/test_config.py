"""Tests for Acceleration-Volatility Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.accel_vol_trend.config import AccelVolTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestAccelVolTrendConfig:
    def test_default_values(self) -> None:
        config = AccelVolTrendConfig()
        assert config.accel_fast == 5
        assert config.accel_slow == 21
        assert config.gk_window == 21
        assert config.accel_smooth == 10
        assert config.accel_long_threshold == 0.005
        assert config.accel_short_threshold == -0.005
        assert config.momentum_window == 21
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = AccelVolTrendConfig()
        with pytest.raises(ValidationError):
            config.accel_fast = 999  # type: ignore[misc]

    def test_accel_fast_range(self) -> None:
        with pytest.raises(ValidationError):
            AccelVolTrendConfig(accel_fast=1)
        with pytest.raises(ValidationError):
            AccelVolTrendConfig(accel_fast=30)

    def test_accel_slow_range(self) -> None:
        with pytest.raises(ValidationError):
            AccelVolTrendConfig(accel_slow=5)
        with pytest.raises(ValidationError):
            AccelVolTrendConfig(accel_slow=100)

    def test_gk_window_range(self) -> None:
        with pytest.raises(ValidationError):
            AccelVolTrendConfig(gk_window=3)
        with pytest.raises(ValidationError):
            AccelVolTrendConfig(gk_window=200)

    def test_accel_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            AccelVolTrendConfig(accel_fast=21, accel_slow=21)
        with pytest.raises(ValidationError):
            AccelVolTrendConfig(accel_fast=21, accel_slow=15)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            AccelVolTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = AccelVolTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.accel_slow + config.accel_smooth

    def test_annualization_factor(self) -> None:
        config = AccelVolTrendConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = AccelVolTrendConfig(accel_fast=3, accel_slow=30)
        assert config.accel_fast == 3
        assert config.accel_slow == 30
