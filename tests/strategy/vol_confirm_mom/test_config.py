"""Tests for Volume-Confirmed Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vol_confirm_mom.config import ShortMode, VolConfirmMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVolConfirmMomConfig:
    def test_default_values(self) -> None:
        config = VolConfirmMomConfig()
        assert config.mom_lookback == 30
        assert config.vol_short_window == 10
        assert config.vol_long_window == 40
        assert config.vol_ratio_clip == 2.0
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VolConfirmMomConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 999  # type: ignore[misc]

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            VolConfirmMomConfig(mom_lookback=2)
        with pytest.raises(ValidationError):
            VolConfirmMomConfig(mom_lookback=200)

    def test_vol_windows_cross_validation(self) -> None:
        """vol_long_window must be > vol_short_window."""
        with pytest.raises(ValidationError):
            VolConfirmMomConfig(vol_short_window=20, vol_long_window=10)
        with pytest.raises(ValidationError):
            VolConfirmMomConfig(vol_short_window=20, vol_long_window=20)

    def test_vol_ratio_clip_range(self) -> None:
        with pytest.raises(ValidationError):
            VolConfirmMomConfig(vol_ratio_clip=0.5)
        with pytest.raises(ValidationError):
            VolConfirmMomConfig(vol_ratio_clip=6.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VolConfirmMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VolConfirmMomConfig()
        assert config.warmup_periods() >= config.vol_long_window
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = VolConfirmMomConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = VolConfirmMomConfig(mom_lookback=50, vol_short_window=15, vol_long_window=60)
        assert config.mom_lookback == 50
        assert config.vol_short_window == 15
        assert config.vol_long_window == 60

    def test_hedge_params(self) -> None:
        config = VolConfirmMomConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
