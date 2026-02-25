"""Tests for KAMA Efficiency Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.kama_eff_trend.config import KamaEffTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestKamaEffTrendConfig:
    def test_default_values(self) -> None:
        config = KamaEffTrendConfig()
        assert config.er_period == 10
        assert config.kama_period == 10
        assert config.kama_fast == 2
        assert config.kama_slow == 30
        assert config.er_threshold == 0.30
        assert config.slope_window == 5
        assert config.vol_target == 0.35
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = KamaEffTrendConfig()
        with pytest.raises(ValidationError):
            config.er_period = 999  # type: ignore[misc]

    def test_er_period_range(self) -> None:
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(er_period=2)
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(er_period=51)

    def test_er_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(er_threshold=0.04)
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(er_threshold=0.81)

    def test_kama_slow_gt_kama_fast(self) -> None:
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(kama_fast=10, kama_slow=10)
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(kama_fast=10, kama_slow=5)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = KamaEffTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.kama_slow + config.slope_window

    def test_annualization_factor(self) -> None:
        config = KamaEffTrendConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = KamaEffTrendConfig(er_period=15, er_threshold=0.40)
        assert config.er_period == 15
        assert config.er_threshold == 0.40

    def test_slope_window_range(self) -> None:
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(slope_window=1)
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(slope_window=21)

    def test_kama_period_range(self) -> None:
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(kama_period=2)
        with pytest.raises(ValidationError):
            KamaEffTrendConfig(kama_period=51)
