"""Tests for EnhancedTSMOMConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.enhanced_tsmom.config import EnhancedTSMOMConfig, ShortMode


class TestEnhancedTSMOMConfig:
    """EnhancedTSMOMConfig 테스트."""

    def test_default_values(self):
        """기본값이 올바르게 설정되는지 확인."""
        config = EnhancedTSMOMConfig()

        assert config.lookback == 30
        assert config.vol_window == 30
        assert config.vol_target == 0.30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.volume_lookback == 20
        assert config.volume_clip_max == 5.0
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8
        assert config.atr_period == 14

    def test_frozen_model(self):
        """frozen 모델이므로 속성 수정 불가."""
        config = EnhancedTSMOMConfig()

        with pytest.raises(ValidationError):
            config.lookback = 60  # type: ignore[misc]

    def test_volume_lookback_range(self):
        """volume_lookback 범위 검증 (5-100)."""
        EnhancedTSMOMConfig(volume_lookback=5)
        EnhancedTSMOMConfig(volume_lookback=100)

        with pytest.raises(ValidationError):
            EnhancedTSMOMConfig(volume_lookback=4)

        with pytest.raises(ValidationError):
            EnhancedTSMOMConfig(volume_lookback=101)

    def test_volume_clip_max_range(self):
        """volume_clip_max 범위 검증 (1.0-20.0)."""
        EnhancedTSMOMConfig(volume_clip_max=1.0)
        EnhancedTSMOMConfig(volume_clip_max=20.0)

        with pytest.raises(ValidationError):
            EnhancedTSMOMConfig(volume_clip_max=0.9)

        with pytest.raises(ValidationError):
            EnhancedTSMOMConfig(volume_clip_max=20.1)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        EnhancedTSMOMConfig(vol_target=0.20, min_volatility=0.05)
        EnhancedTSMOMConfig(vol_target=0.05, min_volatility=0.05)

        with pytest.raises(ValidationError):
            EnhancedTSMOMConfig(vol_target=0.04, min_volatility=0.05)

    def test_warmup_periods(self):
        """warmup_periods가 올바른 워밍업 기간을 반환."""
        config = EnhancedTSMOMConfig(lookback=30, vol_window=30, volume_lookback=20, atr_period=14)
        # evw_warmup = 1 + 20 + 30 = 51, vol_warmup = 1 + 30 = 31, atr = 14
        # max(51, 31, 14) + 1 = 52
        assert config.warmup_periods() == 52

        config = EnhancedTSMOMConfig(lookback=10, vol_window=10, volume_lookback=50, atr_period=14)
        # evw_warmup = 1 + 50 + 10 = 61, vol_warmup = 1 + 10 = 11, atr = 14
        # max(61, 11, 14) + 1 = 62
        assert config.warmup_periods() == 62

    def test_conservative_preset(self):
        """conservative() 프리셋 검증."""
        config = EnhancedTSMOMConfig.conservative()

        assert config.lookback == 48
        assert config.vol_window == 48
        assert config.vol_target == 0.10
        assert config.min_volatility == 0.08
        assert config.volume_lookback == 30
        assert config.volume_clip_max == 3.0

    def test_aggressive_preset(self):
        """aggressive() 프리셋 검증."""
        config = EnhancedTSMOMConfig.aggressive()

        assert config.lookback == 12
        assert config.vol_window == 12
        assert config.vol_target == 0.20
        assert config.min_volatility == 0.05
        assert config.volume_lookback == 10
        assert config.volume_clip_max == 8.0

    def test_for_timeframe_daily(self):
        """for_timeframe('1d') 검증."""
        config = EnhancedTSMOMConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0
        assert config.lookback == 30

    def test_for_timeframe_hourly(self):
        """for_timeframe('1h') 검증."""
        config = EnhancedTSMOMConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0
        assert config.lookback == 24

    def test_for_timeframe_with_overrides(self):
        """for_timeframe에 kwargs 오버라이드."""
        config = EnhancedTSMOMConfig.for_timeframe("4h", vol_target=0.40)
        assert config.annualization_factor == 2190.0
        assert config.vol_target == 0.40

    def test_lookback_range(self):
        """lookback 범위 검증 (6-365)."""
        EnhancedTSMOMConfig(lookback=6)
        EnhancedTSMOMConfig(lookback=365)

        with pytest.raises(ValidationError):
            EnhancedTSMOMConfig(lookback=5)

        with pytest.raises(ValidationError):
            EnhancedTSMOMConfig(lookback=366)


class TestShortMode:
    """ShortMode enum 테스트."""

    def test_values(self):
        """ShortMode 값 검증."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode 사용 가능."""
        config = EnhancedTSMOMConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = EnhancedTSMOMConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED
