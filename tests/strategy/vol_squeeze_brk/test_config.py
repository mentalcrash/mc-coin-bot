"""Tests for Vol Squeeze Breakout config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vol_squeeze_brk.config import ShortMode, VolSqueezeBrkConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVolSqueezeBrkConfig:
    def test_default_values(self) -> None:
        config = VolSqueezeBrkConfig()
        assert config.bb_period == 20
        assert config.bb_std == 2.0
        assert config.bb_pct_window == 120
        assert config.bb_pct_threshold == 0.20
        assert config.atr_period == 14
        assert config.atr_ratio_window == 90
        assert config.atr_ratio_threshold == 0.70
        assert config.vol_surge_window == 42
        assert config.vol_surge_multiplier == 1.3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VolSqueezeBrkConfig()
        with pytest.raises(ValidationError):
            config.bb_period = 999  # type: ignore[misc]

    def test_bb_period_range(self) -> None:
        with pytest.raises(ValidationError):
            VolSqueezeBrkConfig(bb_period=4)
        with pytest.raises(ValidationError):
            VolSqueezeBrkConfig(bb_period=51)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VolSqueezeBrkConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VolSqueezeBrkConfig()
        assert config.warmup_periods() >= config.bb_pct_window

    def test_annualization_factor(self) -> None:
        config = VolSqueezeBrkConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = VolSqueezeBrkConfig(bb_period=30)
        assert config.bb_period == 30
