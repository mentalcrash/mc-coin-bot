"""Tests for Participation Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.participation_mom_12h.config import ParticipationMomConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestParticipationMomConfig:
    def test_default_values(self) -> None:
        config = ParticipationMomConfig()
        assert config.mom_lookback == 20
        assert config.intensity_zscore_window == 30
        assert config.intensity_long_z == 0.5
        assert config.intensity_short_z == -0.5
        assert config.mom_ema_fast == 12
        assert config.mom_ema_slow == 26
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = ParticipationMomConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 999  # type: ignore[misc]

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            ParticipationMomConfig(mom_lookback=4)
        with pytest.raises(ValidationError):
            ParticipationMomConfig(mom_lookback=101)

    def test_intensity_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            ParticipationMomConfig(intensity_zscore_window=9)
        with pytest.raises(ValidationError):
            ParticipationMomConfig(intensity_zscore_window=121)

    def test_intensity_long_z_range(self) -> None:
        with pytest.raises(ValidationError):
            ParticipationMomConfig(intensity_long_z=-0.1)
        with pytest.raises(ValidationError):
            ParticipationMomConfig(intensity_long_z=3.1)

    def test_intensity_short_z_range(self) -> None:
        with pytest.raises(ValidationError):
            ParticipationMomConfig(intensity_short_z=0.1)
        with pytest.raises(ValidationError):
            ParticipationMomConfig(intensity_short_z=-3.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            ParticipationMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_mom_ema_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            ParticipationMomConfig(mom_ema_fast=26, mom_ema_slow=26)
        with pytest.raises(ValidationError):
            ParticipationMomConfig(mom_ema_fast=30, mom_ema_slow=26)

    def test_warmup_periods(self) -> None:
        config = ParticipationMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.intensity_zscore_window
        assert config.warmup_periods() >= config.mom_ema_slow

    def test_annualization_factor(self) -> None:
        config = ParticipationMomConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = ParticipationMomConfig(mom_lookback=30, intensity_long_z=1.0)
        assert config.mom_lookback == 30
        assert config.intensity_long_z == 1.0

    def test_hedge_params_defaults(self) -> None:
        config = ParticipationMomConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
        assert config.atr_period == 14
