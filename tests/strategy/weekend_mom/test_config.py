"""Tests for Weekend-Momentum config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.weekend_mom.config import ShortMode, WeekendMomConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestWeekendMomConfig:
    def test_default_values(self) -> None:
        config = WeekendMomConfig()
        assert config.fast_lookback == 20
        assert config.slow_lookback == 60
        assert config.weekend_boost == 2.0
        assert config.mom_threshold == 0.0
        assert config.short_mom_threshold == 0.0
        assert config.vol_target == 0.35
        assert config.vol_window == 30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 730.0
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen(self) -> None:
        config = WeekendMomConfig()
        with pytest.raises(ValidationError):
            config.fast_lookback = 999  # type: ignore[misc]

    def test_fast_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            WeekendMomConfig(fast_lookback=4)
        with pytest.raises(ValidationError):
            WeekendMomConfig(fast_lookback=101)

    def test_slow_lookback_range(self) -> None:
        with pytest.raises(ValidationError):
            WeekendMomConfig(slow_lookback=19)
        with pytest.raises(ValidationError):
            WeekendMomConfig(slow_lookback=201)

    def test_weekend_boost_range(self) -> None:
        with pytest.raises(ValidationError):
            WeekendMomConfig(weekend_boost=0.5)
        with pytest.raises(ValidationError):
            WeekendMomConfig(weekend_boost=6.0)

    def test_mom_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            WeekendMomConfig(mom_threshold=-0.1)
        with pytest.raises(ValidationError):
            WeekendMomConfig(mom_threshold=0.6)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            WeekendMomConfig(vol_target=0.01, min_volatility=0.05)

    def test_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            WeekendMomConfig(fast_lookback=60, slow_lookback=60)
        with pytest.raises(ValidationError):
            WeekendMomConfig(fast_lookback=80, slow_lookback=60)

    def test_warmup_periods(self) -> None:
        config = WeekendMomConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.slow_lookback

    def test_annualization_factor(self) -> None:
        config = WeekendMomConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = WeekendMomConfig(fast_lookback=10, slow_lookback=40)
        assert config.fast_lookback == 10
        assert config.slow_lookback == 40

    def test_weekend_boost_default_gt_1(self) -> None:
        """Default weekend_boost > 1 confirms weekend weighting."""
        config = WeekendMomConfig()
        assert config.weekend_boost > 1.0

    def test_all_short_modes(self) -> None:
        """All ShortMode values are accepted."""
        for mode in ShortMode:
            config = WeekendMomConfig(short_mode=mode)
            assert config.short_mode == mode
