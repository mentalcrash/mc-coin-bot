"""Tests for vol-compress-brk config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vol_compress_brk.config import ShortMode, VolCompressBrkConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVolCompressBrkConfig:
    def test_default_values(self) -> None:
        config = VolCompressBrkConfig()
        assert config.atr_fast == 7
        assert config.vol_target == 0.35
        assert config.annualization_factor == 35040.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VolCompressBrkConfig()
        with pytest.raises(ValidationError):
            config.atr_fast = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VolCompressBrkConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VolCompressBrkConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = VolCompressBrkConfig()
        assert config.annualization_factor == 35040.0

    def test_custom_params(self) -> None:
        config = VolCompressBrkConfig(atr_fast=5)
        assert config.atr_fast == 5
