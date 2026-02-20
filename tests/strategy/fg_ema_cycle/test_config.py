"""Tests for FgEmaCycle config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fg_ema_cycle.config import FgEmaCycleConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2


class TestFgEmaCycleConfig:
    def test_default_values(self) -> None:
        config = FgEmaCycleConfig()
        assert config.ema_slow_span == 168
        assert config.ema_fast_span == 42
        assert config.fear_cycle == 35.0
        assert config.greed_cycle == 65.0
        assert config.vol_target == 0.35
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = FgEmaCycleConfig()
        with pytest.raises(ValidationError):
            config.ema_slow_span = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FgEmaCycleConfig(vol_target=0.01, min_volatility=0.05)

    def test_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            FgEmaCycleConfig(ema_fast_span=200, ema_slow_span=100)

    def test_fear_lt_greed(self) -> None:
        with pytest.raises(ValidationError):
            FgEmaCycleConfig(fear_cycle=70.0, greed_cycle=30.0)

    def test_warmup_periods(self) -> None:
        config = FgEmaCycleConfig()
        assert config.warmup_periods() >= config.ema_slow_span

    def test_custom_params(self) -> None:
        config = FgEmaCycleConfig(ema_slow_span=120, ema_fast_span=30)
        assert config.ema_slow_span == 120
