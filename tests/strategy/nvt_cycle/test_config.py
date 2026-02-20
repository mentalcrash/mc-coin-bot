"""Tests for NvtCycle config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.nvt_cycle.config import NvtCycleConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestNvtCycleConfig:
    def test_default_values(self) -> None:
        config = NvtCycleConfig()
        assert config.nvt_window == 14
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = NvtCycleConfig()
        with pytest.raises(ValidationError):
            config.nvt_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            NvtCycleConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = NvtCycleConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = NvtCycleConfig(nvt_window=20)
        assert config.nvt_window == 20
