"""Tests for rv-jump-cont config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.rv_jump_cont.config import RvJumpContConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestRvJumpContConfig:
    def test_default_values(self) -> None:
        config = RvJumpContConfig()
        assert config.rv_window == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 35040.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = RvJumpContConfig()
        with pytest.raises(ValidationError):
            config.rv_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            RvJumpContConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = RvJumpContConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = RvJumpContConfig()
        assert config.annualization_factor == 35040.0

    def test_custom_params(self) -> None:
        config = RvJumpContConfig(rv_window=15)
        assert config.rv_window == 15
