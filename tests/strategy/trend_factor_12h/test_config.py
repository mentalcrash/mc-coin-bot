"""Tests for Trend Factor Multi-Horizon config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.trend_factor_12h.config import ShortMode, TrendFactorConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestTrendFactorConfig:
    def test_default_values(self) -> None:
        config = TrendFactorConfig()
        assert config.horizon_1 == 5
        assert config.horizon_2 == 10
        assert config.horizon_3 == 20
        assert config.horizon_4 == 40
        assert config.horizon_5 == 80
        assert config.entry_threshold == 0.5
        assert config.tanh_scale == 0.3
        assert config.vol_target == 0.35
        assert config.vol_window == 30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 730.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = TrendFactorConfig()
        with pytest.raises(ValidationError):
            config.horizon_1 = 999  # type: ignore[misc]

    def test_horizon_1_range(self) -> None:
        with pytest.raises(ValidationError):
            TrendFactorConfig(horizon_1=1)  # too low
        with pytest.raises(ValidationError):
            TrendFactorConfig(horizon_1=31)  # too high

    def test_horizon_5_range(self) -> None:
        with pytest.raises(ValidationError):
            TrendFactorConfig(horizon_5=19)  # too low
        with pytest.raises(ValidationError):
            TrendFactorConfig(horizon_5=401)  # too high

    def test_entry_threshold_range(self) -> None:
        # Lower bound is 0 (valid)
        config = TrendFactorConfig(entry_threshold=0.0)
        assert config.entry_threshold == 0.0
        # Upper bound
        with pytest.raises(ValidationError):
            TrendFactorConfig(entry_threshold=5.1)

    def test_tanh_scale_range(self) -> None:
        with pytest.raises(ValidationError):
            TrendFactorConfig(tanh_scale=0.0)  # gt=0.0
        with pytest.raises(ValidationError):
            TrendFactorConfig(tanh_scale=2.1)  # le=2.0

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            TrendFactorConfig(vol_target=0.01, min_volatility=0.05)

    def test_horizons_strictly_increasing(self) -> None:
        # Equal horizons
        with pytest.raises(ValidationError):
            TrendFactorConfig(horizon_1=10, horizon_2=10)
        # Decreasing horizons
        with pytest.raises(ValidationError):
            TrendFactorConfig(horizon_3=20, horizon_4=15)

    def test_warmup_periods(self) -> None:
        config = TrendFactorConfig()
        assert config.warmup_periods() >= config.horizon_5
        assert config.warmup_periods() >= config.vol_window

    def test_annualization_factor(self) -> None:
        config = TrendFactorConfig()
        assert config.annualization_factor == 730.0

    def test_custom_params(self) -> None:
        config = TrendFactorConfig(horizon_1=3, horizon_2=8, horizon_3=15)
        assert config.horizon_1 == 3
        assert config.horizon_2 == 8
        assert config.horizon_3 == 15

    def test_custom_short_mode(self) -> None:
        config = TrendFactorConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED

    def test_hedge_parameters(self) -> None:
        config = TrendFactorConfig(
            hedge_threshold=-0.10,
            hedge_strength_ratio=0.5,
        )
        assert config.hedge_threshold == -0.10
        assert config.hedge_strength_ratio == 0.5

    def test_hedge_threshold_le_zero(self) -> None:
        with pytest.raises(ValidationError):
            TrendFactorConfig(hedge_threshold=0.01)

    def test_hedge_strength_ratio_range(self) -> None:
        with pytest.raises(ValidationError):
            TrendFactorConfig(hedge_strength_ratio=0.0)
        with pytest.raises(ValidationError):
            TrendFactorConfig(hedge_strength_ratio=1.1)
