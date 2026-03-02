"""Tests for LR-Channel Multi-Scale Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.lr_channel_trend.config import LrChannelTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestLrChannelTrendConfig:
    def test_default_values(self) -> None:
        config = LrChannelTrendConfig()
        assert config.scale_short == 20
        assert config.scale_mid == 60
        assert config.scale_long == 150
        assert config.channel_multiplier == 2.0
        assert config.entry_threshold == 0.22
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen(self) -> None:
        config = LrChannelTrendConfig()
        with pytest.raises(ValidationError):
            config.scale_short = 999  # type: ignore[misc]

    def test_scale_short_range_too_low(self) -> None:
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(scale_short=4, scale_mid=60, scale_long=150)

    def test_scale_short_range_too_high(self) -> None:
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(scale_short=101, scale_mid=200, scale_long=400)

    def test_scale_mid_range_too_low(self) -> None:
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(scale_short=5, scale_mid=9, scale_long=150)

    def test_scale_mid_range_too_high(self) -> None:
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(scale_short=20, scale_mid=301, scale_long=400)

    def test_scale_long_range_too_low(self) -> None:
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(scale_short=5, scale_mid=10, scale_long=19)

    def test_scale_long_range_too_high(self) -> None:
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(scale_short=20, scale_mid=60, scale_long=501)

    def test_channel_multiplier_range_too_low(self) -> None:
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(channel_multiplier=0.4)

    def test_channel_multiplier_range_too_high(self) -> None:
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(channel_multiplier=5.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_vol_target_equal_min_volatility(self) -> None:
        config = LrChannelTrendConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target == config.min_volatility

    def test_scale_ordering_invalid(self) -> None:
        """scale_short < scale_mid < scale_long 위반 시 ValidationError."""
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(scale_short=60, scale_mid=60, scale_long=150)
        with pytest.raises(ValidationError):
            LrChannelTrendConfig(scale_short=20, scale_mid=150, scale_long=100)

    def test_scale_ordering_valid(self) -> None:
        config = LrChannelTrendConfig(scale_short=10, scale_mid=50, scale_long=200)
        assert config.scale_short < config.scale_mid < config.scale_long

    def test_warmup_periods(self) -> None:
        config = LrChannelTrendConfig()
        assert config.warmup_periods() >= config.scale_long

    def test_warmup_periods_custom(self) -> None:
        config = LrChannelTrendConfig(scale_short=10, scale_mid=50, scale_long=200)
        assert config.warmup_periods() >= 200

    def test_custom_params(self) -> None:
        config = LrChannelTrendConfig(
            scale_short=10, scale_mid=40, scale_long=100, channel_multiplier=1.5
        )
        assert config.scale_short == 10
        assert config.channel_multiplier == 1.5
