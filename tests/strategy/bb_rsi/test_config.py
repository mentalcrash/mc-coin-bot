"""Tests for BBRSIConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.bb_rsi.config import BBRSIConfig, ShortMode


class TestBBRSIConfig:
    """BBRSIConfig 테스트."""

    def test_default_values(self):
        """기본값이 올바르게 설정되는지 확인."""
        config = BBRSIConfig()

        assert config.bb_period == 20
        assert config.bb_std == 2.0
        assert config.rsi_period == 14
        assert config.rsi_oversold == 30.0
        assert config.rsi_overbought == 70.0
        assert config.bb_weight == 0.6
        assert config.rsi_weight == 0.4
        assert config.vol_target == 0.20
        assert config.use_adx_filter is True
        assert config.adx_threshold == 25.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen_model(self):
        """frozen 모델이므로 속성 수정 불가."""
        config = BBRSIConfig()

        with pytest.raises(ValidationError):
            config.bb_period = 30  # type: ignore[misc]

    def test_bb_period_range(self):
        """bb_period 범위 검증 (5-100)."""
        BBRSIConfig(bb_period=5)
        BBRSIConfig(bb_period=100)

        with pytest.raises(ValidationError):
            BBRSIConfig(bb_period=4)

        with pytest.raises(ValidationError):
            BBRSIConfig(bb_period=101)

    def test_bb_std_range(self):
        """bb_std 범위 검증 (1.0-4.0)."""
        BBRSIConfig(bb_std=1.0)
        BBRSIConfig(bb_std=4.0)

        with pytest.raises(ValidationError):
            BBRSIConfig(bb_std=0.9)

        with pytest.raises(ValidationError):
            BBRSIConfig(bb_std=4.1)

    def test_rsi_oversold_less_than_overbought(self):
        """rsi_oversold < rsi_overbought 검증."""
        BBRSIConfig(rsi_oversold=25.0, rsi_overbought=75.0)

        with pytest.raises(ValidationError):
            BBRSIConfig(rsi_oversold=70.0, rsi_overbought=30.0)

        with pytest.raises(ValidationError):
            BBRSIConfig(rsi_oversold=50.0, rsi_overbought=50.0)

    def test_weight_sum_must_equal_one(self):
        """bb_weight + rsi_weight = 1.0 검증."""
        BBRSIConfig(bb_weight=0.5, rsi_weight=0.5)
        BBRSIConfig(bb_weight=0.7, rsi_weight=0.3)
        BBRSIConfig(bb_weight=1.0, rsi_weight=0.0)

        with pytest.raises(ValidationError):
            BBRSIConfig(bb_weight=0.5, rsi_weight=0.4)

        with pytest.raises(ValidationError):
            BBRSIConfig(bb_weight=0.8, rsi_weight=0.8)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        BBRSIConfig(vol_target=0.20, min_volatility=0.05)
        BBRSIConfig(vol_target=0.05, min_volatility=0.05)

        with pytest.raises(ValidationError):
            BBRSIConfig(vol_target=0.04, min_volatility=0.05)

    def test_warmup_periods(self):
        """warmup_periods가 최대 기간 + 1을 반환."""
        config = BBRSIConfig(bb_period=30, rsi_period=14, vol_window=20)
        assert config.warmup_periods() == 31

        config = BBRSIConfig(bb_period=10, rsi_period=50, vol_window=20)
        assert config.warmup_periods() == 51

    def test_warmup_periods_includes_adx(self):
        """ADX 필터 활성화 시 ADX 기간도 포함."""
        config = BBRSIConfig(
            bb_period=10, rsi_period=10, vol_window=10, adx_period=50, use_adx_filter=True
        )
        assert config.warmup_periods() == 51

    def test_warmup_periods_excludes_adx_when_disabled(self):
        """ADX 필터 비활성화 시 ADX 기간 미포함."""
        config = BBRSIConfig(
            bb_period=20, rsi_period=14, vol_window=30, adx_period=50, use_adx_filter=False
        )
        assert config.warmup_periods() == 31

    def test_conservative_preset(self):
        """conservative() 프리셋 검증."""
        config = BBRSIConfig.conservative()

        assert config.bb_period == 30
        assert config.bb_std == 2.5
        assert config.vol_target == 0.15
        assert config.atr_stop_multiplier == 2.0

    def test_aggressive_preset(self):
        """aggressive() 프리셋 검증."""
        config = BBRSIConfig.aggressive()

        assert config.bb_period == 14
        assert config.bb_std == 1.5
        assert config.vol_target == 0.30
        assert config.atr_stop_multiplier == 1.0

    def test_for_timeframe_daily(self):
        """for_timeframe('1d') 검증."""
        config = BBRSIConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_hourly(self):
        """for_timeframe('1h') 검증."""
        config = BBRSIConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_with_overrides(self):
        """for_timeframe에 kwargs 오버라이드."""
        config = BBRSIConfig.for_timeframe("4h", vol_target=0.30)
        assert config.annualization_factor == 2190.0
        assert config.vol_target == 0.30


class TestShortMode:
    """ShortMode enum 테스트."""

    def test_values(self):
        """ShortMode 값 검증."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode 사용 가능."""
        config = BBRSIConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = BBRSIConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY
