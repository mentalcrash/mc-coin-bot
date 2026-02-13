"""Tests for cascade-mom config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.cascade_mom.config import CascadeMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestCascadeMomConfig:
    def test_default_values(self) -> None:
        config = CascadeMomConfig()
        assert config.min_streak == 3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = CascadeMomConfig()
        with pytest.raises(ValidationError):
            config.min_streak = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            CascadeMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = CascadeMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = CascadeMomConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = CascadeMomConfig(min_streak=4)
        assert config.min_streak == 4
