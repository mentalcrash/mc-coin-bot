"""Tests for FgDelta config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.fg_delta.config import FgDeltaConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestFgDeltaConfig:
    def test_default_values(self) -> None:
        config = FgDeltaConfig()
        assert config.fg_delta_window == 5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = FgDeltaConfig()
        with pytest.raises(ValidationError):
            config.fg_delta_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            FgDeltaConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = FgDeltaConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = FgDeltaConfig(fg_delta_window=10)
        assert config.fg_delta_window == 10
