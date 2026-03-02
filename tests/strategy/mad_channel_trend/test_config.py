"""Tests for MAD-Channel Multi-Scale Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.mad_channel_trend.config import MadChannelTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestMadChannelTrendConfig:
    def test_default_values(self) -> None:
        config = MadChannelTrendConfig()
        assert config.scale_short == 20
        assert config.scale_mid == 60
        assert config.scale_long == 150
        assert config.mad_multiplier == 2.0
        assert config.keltner_multiplier == 1.5
        assert config.entry_threshold == 0.22
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen(self) -> None:
        config = MadChannelTrendConfig()
        with pytest.raises(ValidationError):
            config.scale_short = 999  # type: ignore[misc]

    def test_scale_short_range(self) -> None:
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(scale_short=4)
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(scale_short=101)

    def test_scale_mid_range(self) -> None:
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(scale_mid=9)
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(scale_mid=301)

    def test_scale_long_range(self) -> None:
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(scale_long=19)
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(scale_long=501)

    def test_mad_multiplier_range(self) -> None:
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(mad_multiplier=0.4)
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(mad_multiplier=5.1)

    def test_keltner_multiplier_range(self) -> None:
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(keltner_multiplier=0.4)
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(keltner_multiplier=4.1)

    def test_entry_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(entry_threshold=-0.1)
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(entry_threshold=1.5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_scale_ordering_validation(self) -> None:
        """scale_short < scale_mid < scale_long 필수."""
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(scale_short=60, scale_mid=60, scale_long=150)
        with pytest.raises(ValidationError):
            MadChannelTrendConfig(scale_short=20, scale_mid=150, scale_long=60)

    def test_warmup_periods(self) -> None:
        config = MadChannelTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.scale_long

    def test_custom_params(self) -> None:
        config = MadChannelTrendConfig(scale_short=10, scale_mid=40, scale_long=100)
        assert config.scale_short == 10
        assert config.scale_mid == 40
        assert config.scale_long == 100

    def test_custom_channel_params(self) -> None:
        config = MadChannelTrendConfig(mad_multiplier=1.5, keltner_multiplier=2.0)
        assert config.mad_multiplier == 1.5
        assert config.keltner_multiplier == 2.0
