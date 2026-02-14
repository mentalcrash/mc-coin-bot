"""Tests for Volume MACD Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vmacd_mom.config import ShortMode, VmacdMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2


class TestVmacdMomConfig:
    def test_default_values(self) -> None:
        config = VmacdMomConfig()
        assert config.vmacd_fast == 12
        assert config.vmacd_slow == 26
        assert config.vmacd_signal == 9
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = VmacdMomConfig()
        with pytest.raises(ValidationError):
            config.vmacd_fast = 999  # type: ignore[misc]

    def test_vmacd_slow_must_exceed_fast(self) -> None:
        with pytest.raises(ValidationError):
            VmacdMomConfig(vmacd_fast=26, vmacd_slow=26)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VmacdMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VmacdMomConfig()
        assert config.warmup_periods() >= config.vmacd_slow

    def test_custom_params(self) -> None:
        config = VmacdMomConfig(vmacd_fast=10, vmacd_slow=20, vmacd_signal=7)
        assert config.vmacd_fast == 10
        assert config.vmacd_slow == 20
