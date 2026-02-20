"""Tests for FgPersistBreak config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fg_persist_break.config import FgPersistBreakConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFgPersistBreakConfig:
    def test_default_values(self) -> None:
        config = FgPersistBreakConfig()
        assert config.fear_threshold == 25.0
        assert config.greed_threshold == 75.0
        assert config.min_persist == 5
        assert config.max_streak_cap == 20
        assert config.price_mom_window == 5
        assert config.vol_target == 0.35
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = FgPersistBreakConfig()
        with pytest.raises(ValidationError):
            config.fear_threshold = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FgPersistBreakConfig(vol_target=0.01, min_volatility=0.05)

    def test_fear_lt_greed(self) -> None:
        with pytest.raises(ValidationError):
            FgPersistBreakConfig(fear_threshold=80.0, greed_threshold=20.0)

    def test_warmup_periods(self) -> None:
        config = FgPersistBreakConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = FgPersistBreakConfig(min_persist=10)
        assert config.min_persist == 10
