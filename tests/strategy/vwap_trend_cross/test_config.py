"""Tests for VWAP Trend Crossover config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vwap_trend_cross.config import ShortMode, VwapTrendCrossConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVwapTrendCrossConfig:
    def test_default_values(self) -> None:
        config = VwapTrendCrossConfig()
        assert config.vwap_short_window == 20
        assert config.vwap_long_window == 60
        assert config.spread_clip == 0.05
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = VwapTrendCrossConfig()
        with pytest.raises(ValidationError):
            config.vwap_short_window = 999  # type: ignore[misc]

    def test_vwap_short_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VwapTrendCrossConfig(vwap_short_window=2)
        with pytest.raises(ValidationError):
            VwapTrendCrossConfig(vwap_short_window=200)

    def test_vwap_windows_cross_validation(self) -> None:
        """vwap_long_window must be > vwap_short_window."""
        with pytest.raises(ValidationError):
            VwapTrendCrossConfig(vwap_short_window=30, vwap_long_window=20)
        with pytest.raises(ValidationError):
            VwapTrendCrossConfig(vwap_short_window=30, vwap_long_window=30)

    def test_spread_clip_range(self) -> None:
        with pytest.raises(ValidationError):
            VwapTrendCrossConfig(spread_clip=0.001)
        with pytest.raises(ValidationError):
            VwapTrendCrossConfig(spread_clip=0.5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VwapTrendCrossConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = VwapTrendCrossConfig()
        assert config.warmup_periods() >= config.vwap_long_window
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = VwapTrendCrossConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = VwapTrendCrossConfig(vwap_short_window=30, vwap_long_window=90)
        assert config.vwap_short_window == 30
        assert config.vwap_long_window == 90

    def test_hedge_params(self) -> None:
        config = VwapTrendCrossConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
