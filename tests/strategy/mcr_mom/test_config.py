"""Tests for Momentum Crash Filter config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.mcr_mom.config import McrMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2


class TestMcrMomConfig:
    def test_default_values(self) -> None:
        config = McrMomConfig()
        assert config.mom_lookback == 30
        assert config.vov_window == 20
        assert config.vov_crash_threshold == 0.8
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = McrMomConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 999  # type: ignore[misc]

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            McrMomConfig(mom_lookback=4)
        with pytest.raises(ValidationError):
            McrMomConfig(mom_lookback=121)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            McrMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = McrMomConfig()
        assert config.warmup_periods() >= config.vol_window

    def test_custom_params(self) -> None:
        config = McrMomConfig(mom_lookback=20, vov_window=15)
        assert config.mom_lookback == 20
        assert config.vov_window == 15
