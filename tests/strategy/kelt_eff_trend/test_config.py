"""Tests for Keltner Efficiency Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.kelt_eff_trend.config import KeltEffTrendConfig, ShortMode


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestKeltEffTrendConfig:
    def test_default_values(self) -> None:
        config = KeltEffTrendConfig()
        assert config.kc_ema_period == 20
        assert config.kc_atr_period == 10
        assert config.kc_multiplier == 1.5
        assert config.er_period == 10
        assert config.er_threshold == 0.3
        assert config.vol_target == 0.35
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen(self) -> None:
        config = KeltEffTrendConfig()
        with pytest.raises(ValidationError):
            config.kc_ema_period = 999  # type: ignore[misc]

    def test_kc_ema_period_range(self) -> None:
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(kc_ema_period=4)
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(kc_ema_period=101)

    def test_kc_atr_period_range(self) -> None:
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(kc_atr_period=2)
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(kc_atr_period=51)

    def test_kc_multiplier_range(self) -> None:
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(kc_multiplier=0.4)
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(kc_multiplier=5.1)

    def test_er_period_range(self) -> None:
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(er_period=2)
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(er_period=51)

    def test_er_threshold_range(self) -> None:
        config = KeltEffTrendConfig(er_threshold=0.0)
        assert config.er_threshold == 0.0
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(er_threshold=-0.1)
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(er_threshold=1.1)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            KeltEffTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_warmup_periods(self) -> None:
        config = KeltEffTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.kc_ema_period

    def test_annualization_factor(self) -> None:
        config = KeltEffTrendConfig()
        assert config.annualization_factor == 2190.0

    def test_custom_params(self) -> None:
        config = KeltEffTrendConfig(kc_ema_period=30, er_threshold=0.5)
        assert config.kc_ema_period == 30
        assert config.er_threshold == 0.5

    def test_hedge_params(self) -> None:
        config = KeltEffTrendConfig()
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
