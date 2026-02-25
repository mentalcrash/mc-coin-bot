"""Tests for Multi-Source Directional Composite config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.multi_source_composite.config import MultiSourceCompositeConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestMultiSourceCompositeConfig:
    def test_default_values(self) -> None:
        config = MultiSourceCompositeConfig()
        assert config.mom_fast == 10
        assert config.mom_slow == 30
        assert config.mom_lookback == 20
        assert config.velocity_fast_window == 7
        assert config.velocity_slow_window == 30
        assert config.fg_delta_window == 7
        assert config.fg_smooth_window == 5
        assert config.fg_threshold == 2.0
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = MultiSourceCompositeConfig()
        with pytest.raises(ValidationError):
            config.mom_fast = 999  # type: ignore[misc]

    def test_mom_fast_range(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(mom_fast=2)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(mom_fast=51)

    def test_mom_slow_range(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(mom_slow=9)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(mom_slow=121)

    def test_mom_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(mom_lookback=4)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(mom_lookback=121)

    def test_velocity_fast_window_range(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(velocity_fast_window=2)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(velocity_fast_window=31)

    def test_velocity_slow_window_range(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(velocity_slow_window=9)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(velocity_slow_window=121)

    def test_fg_delta_window_range(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(fg_delta_window=2)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(fg_delta_window=31)

    def test_fg_smooth_window_range(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(fg_smooth_window=2)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(fg_smooth_window=21)

    def test_fg_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(fg_threshold=-1.0)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(fg_threshold=21.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(vol_target=0.01, min_volatility=0.05)

    def test_mom_fast_lt_mom_slow(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(mom_fast=30, mom_slow=30)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(mom_fast=40, mom_slow=30)

    def test_velocity_fast_lt_velocity_slow(self) -> None:
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(velocity_fast_window=30, velocity_slow_window=30)
        with pytest.raises(ValidationError):
            MultiSourceCompositeConfig(velocity_fast_window=30, velocity_slow_window=20)

    def test_warmup_periods(self) -> None:
        config = MultiSourceCompositeConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.mom_slow
        assert config.warmup_periods() >= config.velocity_slow_window

    def test_annualization_factor(self) -> None:
        config = MultiSourceCompositeConfig()
        assert config.annualization_factor == 365.0

    def test_custom_params(self) -> None:
        config = MultiSourceCompositeConfig(mom_fast=5, mom_slow=50, fg_threshold=3.0)
        assert config.mom_fast == 5
        assert config.mom_slow == 50
        assert config.fg_threshold == 3.0

    def test_hedge_only_params(self) -> None:
        config = MultiSourceCompositeConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5
