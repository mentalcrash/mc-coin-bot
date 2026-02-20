"""Tests for EMA Cross Base config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ema_cross_base.config import EmaCrossBaseConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestEmaCrossBaseConfig:
    def test_default_values(self) -> None:
        config = EmaCrossBaseConfig()
        assert config.fast_period == 20
        assert config.slow_period == 100
        assert config.vol_target == 0.35
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = EmaCrossBaseConfig()
        with pytest.raises(ValidationError):
            config.fast_period = 999  # type: ignore[misc]

    def test_fast_period_range(self) -> None:
        with pytest.raises(ValidationError):
            EmaCrossBaseConfig(fast_period=4)
        with pytest.raises(ValidationError):
            EmaCrossBaseConfig(fast_period=51)

    def test_slow_period_range(self) -> None:
        with pytest.raises(ValidationError):
            EmaCrossBaseConfig(slow_period=49)
        with pytest.raises(ValidationError):
            EmaCrossBaseConfig(slow_period=301)

    def test_warmup_periods(self) -> None:
        config = EmaCrossBaseConfig()
        assert config.warmup_periods() >= config.slow_period

    def test_custom_params(self) -> None:
        config = EmaCrossBaseConfig(fast_period=10, slow_period=50)
        assert config.fast_period == 10
        assert config.slow_period == 50
