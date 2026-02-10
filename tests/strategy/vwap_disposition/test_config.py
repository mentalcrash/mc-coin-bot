"""Tests for VWAPDispositionConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.vwap_disposition.config import ShortMode, VWAPDispositionConfig


class TestShortMode:
    def test_values(self):
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        config = VWAPDispositionConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestVWAPDispositionConfig:
    def test_default_values(self):
        config = VWAPDispositionConfig()

        assert config.vwap_window == 720
        assert config.overhang_high == 0.15
        assert config.overhang_low == 0.10
        assert config.vol_ratio_window == 20
        assert config.vol_spike_threshold == 1.5
        assert config.vol_decline_threshold == 0.7
        assert config.use_volume_confirm is True
        assert config.mom_lookback == 20
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        config = VWAPDispositionConfig()
        with pytest.raises(ValidationError):
            config.vwap_window = 500  # type: ignore[misc]

    def test_vwap_window_range(self):
        config = VWAPDispositionConfig(vwap_window=100)
        assert config.vwap_window == 100

        config = VWAPDispositionConfig(vwap_window=2000)
        assert config.vwap_window == 2000

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(vwap_window=99)

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(vwap_window=2001)

    def test_overhang_high_range(self):
        config = VWAPDispositionConfig(overhang_high=0.05)
        assert config.overhang_high == 0.05

        config = VWAPDispositionConfig(overhang_high=0.50)
        assert config.overhang_high == 0.50

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(overhang_high=0.04)

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(overhang_high=0.51)

    def test_overhang_low_range(self):
        config = VWAPDispositionConfig(overhang_low=0.03)
        assert config.overhang_low == 0.03

        config = VWAPDispositionConfig(overhang_low=0.30)
        assert config.overhang_low == 0.30

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(overhang_low=0.02)

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(overhang_low=0.31)

    def test_vol_ratio_window_range(self):
        config = VWAPDispositionConfig(vol_ratio_window=5)
        assert config.vol_ratio_window == 5

        config = VWAPDispositionConfig(vol_ratio_window=60)
        assert config.vol_ratio_window == 60

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(vol_ratio_window=4)

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(vol_ratio_window=61)

    def test_vol_spike_threshold_range(self):
        config = VWAPDispositionConfig(vol_spike_threshold=1.0)
        assert config.vol_spike_threshold == 1.0

        config = VWAPDispositionConfig(vol_spike_threshold=3.0)
        assert config.vol_spike_threshold == 3.0

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(vol_spike_threshold=0.9)

    def test_vol_decline_threshold_range(self):
        config = VWAPDispositionConfig(vol_decline_threshold=0.3)
        assert config.vol_decline_threshold == 0.3

        config = VWAPDispositionConfig(vol_decline_threshold=1.0)
        assert config.vol_decline_threshold == 1.0

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(vol_decline_threshold=0.2)

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(vol_decline_threshold=1.1)

    def test_mom_lookback_range(self):
        config = VWAPDispositionConfig(mom_lookback=5)
        assert config.mom_lookback == 5

        config = VWAPDispositionConfig(mom_lookback=60)
        assert config.mom_lookback == 60

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(mom_lookback=4)

    def test_vol_target_gte_min_volatility(self):
        config = VWAPDispositionConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        config = VWAPDispositionConfig(
            vwap_window=720,
            vol_ratio_window=20,
            mom_lookback=20,
            atr_period=14,
        )
        # max(720, 20, 20, 14) + 1 = 721
        assert config.warmup_periods() == 721

    def test_warmup_periods_custom(self):
        config = VWAPDispositionConfig(
            vwap_window=100,
            vol_ratio_window=60,
            mom_lookback=60,
            atr_period=50,
        )
        # max(100, 60, 60, 50) + 1 = 101
        assert config.warmup_periods() == 101

    def test_default_short_mode_is_full(self):
        """기본 short_mode는 FULL이어야 함."""
        config = VWAPDispositionConfig()
        assert config.short_mode == ShortMode.FULL

    def test_atr_period_range(self):
        config = VWAPDispositionConfig(atr_period=5)
        assert config.atr_period == 5

        config = VWAPDispositionConfig(atr_period=50)
        assert config.atr_period == 50

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(atr_period=4)

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(atr_period=51)

    def test_hedge_threshold_range(self):
        config = VWAPDispositionConfig(hedge_threshold=-0.30)
        assert config.hedge_threshold == -0.30

        config = VWAPDispositionConfig(hedge_threshold=-0.05)
        assert config.hedge_threshold == -0.05

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(hedge_threshold=-0.31)

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(hedge_threshold=-0.04)

    def test_hedge_strength_ratio_range(self):
        config = VWAPDispositionConfig(hedge_strength_ratio=0.1)
        assert config.hedge_strength_ratio == 0.1

        config = VWAPDispositionConfig(hedge_strength_ratio=1.0)
        assert config.hedge_strength_ratio == 1.0

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(hedge_strength_ratio=0.09)

        with pytest.raises(ValidationError):
            VWAPDispositionConfig(hedge_strength_ratio=1.1)

    def test_use_volume_confirm_toggle(self):
        config = VWAPDispositionConfig(use_volume_confirm=False)
        assert config.use_volume_confirm is False

        config = VWAPDispositionConfig(use_volume_confirm=True)
        assert config.use_volume_confirm is True
