"""Tests for VRP-Regime Trend config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.vrp_regime_trend.config import ShortMode, VrpRegimeTrendConfig


class TestShortMode:
    def test_values(self) -> None:
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_enum(self) -> None:
        assert isinstance(ShortMode.DISABLED, int)


class TestVrpRegimeTrendConfig:
    def test_default_values(self) -> None:
        config = VrpRegimeTrendConfig()
        assert config.gk_rv_window == 30
        assert config.vrp_ma_window == 14
        assert config.vrp_zscore_window == 90
        assert config.vrp_high_z == 0.5
        assert config.vrp_low_z == -0.5
        assert config.trend_ema_fast == 12
        assert config.trend_ema_slow == 36
        assert config.vol_target == 0.35
        assert config.annualization_factor == 1095.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen(self) -> None:
        config = VrpRegimeTrendConfig()
        with pytest.raises(ValidationError):
            config.gk_rv_window = 999  # type: ignore[misc]

    def test_gk_rv_window_range(self) -> None:
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(gk_rv_window=2)
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(gk_rv_window=200)

    def test_vrp_high_z_range(self) -> None:
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(vrp_high_z=-1.0)
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(vrp_high_z=4.0)

    def test_vrp_low_z_range(self) -> None:
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(vrp_low_z=1.0)
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(vrp_low_z=-4.0)

    def test_vol_target_gte_min_volatility(self) -> None:
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(vol_target=0.01, min_volatility=0.05)

    def test_vrp_high_z_gt_vrp_low_z(self) -> None:
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(vrp_high_z=0.0, vrp_low_z=0.0)
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(vrp_high_z=-0.5, vrp_low_z=0.5)

    def test_trend_ema_fast_lt_slow(self) -> None:
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(trend_ema_fast=50, trend_ema_slow=20)
        with pytest.raises(ValidationError):
            VrpRegimeTrendConfig(trend_ema_fast=20, trend_ema_slow=20)

    def test_warmup_periods(self) -> None:
        config = VrpRegimeTrendConfig()
        assert config.warmup_periods() >= config.vol_window
        assert config.warmup_periods() >= config.trend_ema_slow

    def test_annualization_factor(self) -> None:
        config = VrpRegimeTrendConfig()
        assert config.annualization_factor == 1095.0

    def test_custom_params(self) -> None:
        config = VrpRegimeTrendConfig(gk_rv_window=20, vrp_high_z=1.0)
        assert config.gk_rv_window == 20
        assert config.vrp_high_z == 1.0
