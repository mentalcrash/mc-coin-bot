"""Tests for Return Streak Persistence config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.streak_persistence.config import ShortMode, StreakPersistenceConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestStreakPersistenceConfig:
    def test_default_values(self) -> None:
        config = StreakPersistenceConfig()
        assert config.streak_threshold == 3
        assert config.max_streak_cap == 7
        assert config.momentum_lookback == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = StreakPersistenceConfig()
        with pytest.raises(ValidationError):
            config.streak_threshold = 999  # type: ignore[misc]

    def test_streak_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            StreakPersistenceConfig(streak_threshold=1)
        with pytest.raises(ValidationError):
            StreakPersistenceConfig(streak_threshold=11)

    def test_max_streak_cap_range(self) -> None:
        with pytest.raises(ValidationError):
            StreakPersistenceConfig(max_streak_cap=2)
        with pytest.raises(ValidationError):
            StreakPersistenceConfig(max_streak_cap=16)

    def test_cap_must_be_gte_threshold(self) -> None:
        with pytest.raises(ValidationError):
            StreakPersistenceConfig(streak_threshold=5, max_streak_cap=4)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            StreakPersistenceConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = StreakPersistenceConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = StreakPersistenceConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = StreakPersistenceConfig(streak_threshold=4, max_streak_cap=8)
        assert config.streak_threshold == 4
        assert config.max_streak_cap == 8

    def test_momentum_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            StreakPersistenceConfig(momentum_lookback=3)
        with pytest.raises(ValidationError):
            StreakPersistenceConfig(momentum_lookback=61)
