"""Tests for MVRV Cycle Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.mvrv_cycle_trend.config import MvrvCycleTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestMvrvCycleTrendConfig:
    def test_default_values(self) -> None:
        config = MvrvCycleTrendConfig()
        assert config.mvrv_zscore_window == 365
        assert config.mvrv_bull_threshold == -0.5
        assert config.mvrv_bear_threshold == 2.0
        assert config.mom_fast == 14
        assert config.mom_slow == 60
        assert config.mom_blend_weight == 0.6
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = MvrvCycleTrendConfig()
        with pytest.raises(ValidationError):
            config.mom_fast = 999  # type: ignore[misc]

    def test_mvrv_zscore_window_range(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mvrv_zscore_window=50)  # < 90
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mvrv_zscore_window=800)  # > 730

    def test_mom_fast_range(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mom_fast=1)  # < 3
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mom_fast=100)  # > 60

    def test_mom_slow_range(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mom_slow=10)  # < 20
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mom_slow=200)  # > 180

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_mvrv_bull_lt_bear(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mvrv_bull_threshold=3.0, mvrv_bear_threshold=1.0)

    def test_mvrv_bull_eq_bear(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mvrv_bull_threshold=1.0, mvrv_bear_threshold=1.0)

    def test_mom_fast_lt_mom_slow(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mom_fast=60, mom_slow=30)

    def test_warmup_periods(self) -> None:
        config = MvrvCycleTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.mom_slow

    def test_annualization_factor_12h(self) -> None:
        config = MvrvCycleTrendConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = MvrvCycleTrendConfig(mom_fast=10, mom_slow=40)
        assert config.mom_fast == 10
        assert config.mom_slow == 40

    def test_mom_blend_weight_range(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mom_blend_weight=0.05)  # < 0.1
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(mom_blend_weight=0.95)  # > 0.9

    def test_hedge_threshold_must_be_nonpositive(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(hedge_threshold=0.1)

    def test_hedge_strength_ratio_range(self) -> None:
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(hedge_strength_ratio=0.0)  # must be > 0
        with pytest.raises(ValidationError):
            MvrvCycleTrendConfig(hedge_strength_ratio=1.5)  # must be <= 1.0
