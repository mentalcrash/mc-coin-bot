"""Tests for VRRegimeConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.vr_regime.config import ShortMode, VRRegimeConfig


class TestShortMode:
    def test_values(self):
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        config = VRRegimeConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestVRRegimeConfig:
    def test_default_values(self):
        config = VRRegimeConfig()

        assert config.vr_window == 120
        assert config.vr_k == 5
        assert config.significance_z == 1.96
        assert config.mom_lookback == 20
        assert config.use_heteroscedastic is True
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        config = VRRegimeConfig()
        with pytest.raises(ValidationError):
            config.vr_window = 200  # type: ignore[misc]

    def test_vr_window_range(self):
        config = VRRegimeConfig(vr_window=40)
        assert config.vr_window == 40

        config = VRRegimeConfig(vr_window=500)
        assert config.vr_window == 500

        with pytest.raises(ValidationError):
            VRRegimeConfig(vr_window=39)

        with pytest.raises(ValidationError):
            VRRegimeConfig(vr_window=501)

    def test_vr_k_range(self):
        config = VRRegimeConfig(vr_k=2)
        assert config.vr_k == 2

        config = VRRegimeConfig(vr_k=20, vr_window=500)
        assert config.vr_k == 20

        with pytest.raises(ValidationError):
            VRRegimeConfig(vr_k=1)

        with pytest.raises(ValidationError):
            VRRegimeConfig(vr_k=21)

    def test_vr_k_times_2_lt_window(self):
        """vr_k * 2 < vr_window 검증."""
        config = VRRegimeConfig(vr_k=5, vr_window=120)
        assert config.vr_k * 2 < config.vr_window

        with pytest.raises(ValidationError):
            VRRegimeConfig(vr_k=20, vr_window=40)

        with pytest.raises(ValidationError):
            VRRegimeConfig(vr_k=10, vr_window=20, mom_lookback=5)

    def test_significance_z_range(self):
        config = VRRegimeConfig(significance_z=1.0)
        assert config.significance_z == 1.0

        config = VRRegimeConfig(significance_z=3.0)
        assert config.significance_z == 3.0

        with pytest.raises(ValidationError):
            VRRegimeConfig(significance_z=0.5)

    def test_mom_lookback_range(self):
        config = VRRegimeConfig(mom_lookback=5)
        assert config.mom_lookback == 5

        config = VRRegimeConfig(mom_lookback=60)
        assert config.mom_lookback == 60

        with pytest.raises(ValidationError):
            VRRegimeConfig(mom_lookback=4)

    def test_vol_target_gte_min_volatility(self):
        config = VRRegimeConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            VRRegimeConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        config = VRRegimeConfig(
            vr_window=120,
            mom_lookback=20,
            atr_period=14,
        )
        # max(120, 20, 14) + 1 = 121
        assert config.warmup_periods() == 121

    def test_warmup_periods_custom(self):
        config = VRRegimeConfig(
            vr_window=200,
            mom_lookback=60,
            atr_period=50,
        )
        # max(200, 60, 50) + 1 = 201
        assert config.warmup_periods() == 201
