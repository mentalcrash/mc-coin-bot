"""Tests for VWAP-Channel Multi-Scale config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vwap_channel_12h.config import ShortMode, VwapChannelConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVwapChannelConfig:
    def test_default_values(self) -> None:
        config = VwapChannelConfig()
        assert config.scale_short == 20
        assert config.scale_mid == 60
        assert config.scale_long == 150
        assert config.band_multiplier == 2.0
        assert config.atr_period == 14
        assert config.entry_threshold == 0.33
        assert config.vol_target == 0.35
        assert config.vol_window == 30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen(self) -> None:
        config = VwapChannelConfig()
        with pytest.raises(ValidationError):
            config.scale_short = 999  # type: ignore[misc]

    def test_scale_short_range(self) -> None:
        with pytest.raises(ValidationError):
            VwapChannelConfig(scale_short=4)
        with pytest.raises(ValidationError):
            VwapChannelConfig(scale_short=101)

    def test_scale_mid_range(self) -> None:
        with pytest.raises(ValidationError):
            VwapChannelConfig(scale_mid=9)
        with pytest.raises(ValidationError):
            VwapChannelConfig(scale_mid=301)

    def test_scale_long_range(self) -> None:
        with pytest.raises(ValidationError):
            VwapChannelConfig(scale_long=19)
        with pytest.raises(ValidationError):
            VwapChannelConfig(scale_long=501)

    def test_band_multiplier_range(self) -> None:
        with pytest.raises(ValidationError):
            VwapChannelConfig(band_multiplier=0.4)
        with pytest.raises(ValidationError):
            VwapChannelConfig(band_multiplier=5.1)

    def test_entry_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            VwapChannelConfig(entry_threshold=-0.1)
        with pytest.raises(ValidationError):
            VwapChannelConfig(entry_threshold=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VwapChannelConfig(vol_target=0.01, min_volatility=0.05)

    def test_scale_ordering_violation(self) -> None:
        """scale_short < scale_mid < scale_long 위반 시 에러."""
        with pytest.raises(ValidationError):
            VwapChannelConfig(scale_short=60, scale_mid=60, scale_long=150)
        with pytest.raises(ValidationError):
            VwapChannelConfig(scale_short=20, scale_mid=150, scale_long=60)

    def test_warmup_periods(self) -> None:
        config = VwapChannelConfig()
        wp = config.warmup_periods()
        assert wp >= config.scale_long
        assert wp >= config.vol_window
        assert wp >= config.atr_period

    def test_annualization_factor(self) -> None:
        config = VwapChannelConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = VwapChannelConfig(
            scale_short=10,
            scale_mid=40,
            scale_long=120,
            band_multiplier=1.5,
        )
        assert config.scale_short == 10
        assert config.scale_mid == 40
        assert config.scale_long == 120
        assert config.band_multiplier == 1.5

    def test_hedge_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            VwapChannelConfig(hedge_threshold=0.1)

    def test_hedge_strength_ratio_range(self) -> None:
        with pytest.raises(ValidationError):
            VwapChannelConfig(hedge_strength_ratio=0.0)
        with pytest.raises(ValidationError):
            VwapChannelConfig(hedge_strength_ratio=1.1)
