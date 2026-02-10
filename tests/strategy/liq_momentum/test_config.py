"""Tests for LiqMomentumConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.liq_momentum.config import LiqMomentumConfig, ShortMode


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode를 올바르게 수용."""
        config = LiqMomentumConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL


class TestLiqMomentumConfig:
    """LiqMomentumConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = LiqMomentumConfig()

        assert config.rel_vol_window == 168
        assert config.amihud_window == 24
        assert config.amihud_pctl_window == 720
        assert config.rel_vol_low == 0.5
        assert config.rel_vol_high == 1.5
        assert config.amihud_pctl_high == 0.75
        assert config.amihud_pctl_low == 0.25
        assert config.mom_lookback == 12
        assert config.low_liq_multiplier == 1.5
        assert config.high_liq_multiplier == 0.5
        assert config.weekend_multiplier == 1.2
        assert config.vol_target == 0.30
        assert config.annualization_factor == 8760.0
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = LiqMomentumConfig()

        with pytest.raises(ValidationError):
            config.rel_vol_window = 48  # type: ignore[misc]

    def test_rel_vol_window_range(self):
        """rel_vol_window 범위 검증."""
        config = LiqMomentumConfig(rel_vol_window=24)
        assert config.rel_vol_window == 24

        with pytest.raises(ValidationError):
            LiqMomentumConfig(rel_vol_window=23)

        with pytest.raises(ValidationError):
            LiqMomentumConfig(rel_vol_window=721)

    def test_rel_vol_low_must_be_less_than_high(self):
        """rel_vol_low < rel_vol_high 검증."""
        with pytest.raises(ValidationError):
            LiqMomentumConfig(rel_vol_low=1.5, rel_vol_high=1.5)

        with pytest.raises(ValidationError):
            LiqMomentumConfig(rel_vol_low=2.0, rel_vol_high=1.5)

    def test_amihud_pctl_low_must_be_less_than_high(self):
        """amihud_pctl_low < amihud_pctl_high 검증."""
        with pytest.raises(ValidationError):
            LiqMomentumConfig(amihud_pctl_low=0.75, amihud_pctl_high=0.75)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        with pytest.raises(ValidationError):
            LiqMomentumConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = LiqMomentumConfig()
        # max(720, 168, 14) + 1 = 721
        assert config.warmup_periods() == 721

    def test_warmup_periods_custom(self):
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = LiqMomentumConfig(
            amihud_pctl_window=48,
            rel_vol_window=100,
            atr_period=50,
        )
        # max(48, 100, 50) + 1 = 101
        assert config.warmup_periods() == 101

    def test_mom_lookback_range(self):
        """mom_lookback 범위 검증."""
        config = LiqMomentumConfig(mom_lookback=3)
        assert config.mom_lookback == 3

        with pytest.raises(ValidationError):
            LiqMomentumConfig(mom_lookback=2)

        with pytest.raises(ValidationError):
            LiqMomentumConfig(mom_lookback=169)
