"""Tests for vmsm config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vmsm.config import ShortMode, VmsmConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVmsmConfig:
    def test_default_values(self) -> None:
        config = VmsmConfig()
        assert config.roc_short == 8
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2920.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VmsmConfig()
        with pytest.raises(ValidationError):
            config.roc_short = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VmsmConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VmsmConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = VmsmConfig()
        assert config.annualization_factor == 2920.0

    def test_custom_params(self) -> None:
        config = VmsmConfig(roc_short=6)
        assert config.roc_short == 6
