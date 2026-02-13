"""Tests for vol-impulse-mom config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vol_impulse_mom.config import ShortMode, VolImpulseMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVolImpulseMomConfig:
    def test_default_values(self) -> None:
        config = VolImpulseMomConfig()
        assert config.vol_spike_window == 20
        assert config.vol_target == 0.35
        assert config.annualization_factor == 35040.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VolImpulseMomConfig()
        with pytest.raises(ValidationError):
            config.vol_spike_window = 999  # type: ignore[misc]

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VolImpulseMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VolImpulseMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = VolImpulseMomConfig()
        assert config.annualization_factor == 35040.0

    def test_custom_params(self) -> None:
        config = VolImpulseMomConfig(vol_spike_window=15)
        assert config.vol_spike_window == 15
