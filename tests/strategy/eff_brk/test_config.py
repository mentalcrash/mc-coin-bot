"""Tests for Efficiency Breakout config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.eff_brk.config import EffBrkConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestEffBrkConfig:
    def test_default_values(self) -> None:
        config = EffBrkConfig()
        assert config.er_period == 10
        assert config.er_threshold == 0.35
        assert config.er_exit_threshold == 0.15
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = EffBrkConfig()
        with pytest.raises(ValidationError):
            config.er_period = 999  # type: ignore[misc]

    def test_er_period_range(self) -> None:
        with pytest.raises(ValidationError):
            EffBrkConfig(er_period=2)
        with pytest.raises(ValidationError):
            EffBrkConfig(er_period=61)

    def test_er_threshold_must_exceed_exit(self) -> None:
        with pytest.raises(ValidationError):
            EffBrkConfig(er_threshold=0.15, er_exit_threshold=0.15)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            EffBrkConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = EffBrkConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.er_period

    def test_annualization_factor(self) -> None:
        config = EffBrkConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = EffBrkConfig(er_period=20, er_threshold=0.4, er_exit_threshold=0.2)
        assert config.er_period == 20
        assert config.er_threshold == 0.4
