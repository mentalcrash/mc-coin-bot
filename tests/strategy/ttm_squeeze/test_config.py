"""Tests for TtmSqueezeConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.ttm_squeeze.config import ShortMode, TtmSqueezeConfig


class TestTtmSqueezeConfigDefaults:
    """TtmSqueezeConfig 기본값 테스트."""

    def test_default_values(self) -> None:
        """기본값이 올바르게 설정되는지 확인."""
        config = TtmSqueezeConfig()

        assert config.bb_period == 20
        assert config.bb_std == 2.0
        assert config.kc_period == 20
        assert config.kc_mult == 1.5
        assert config.mom_period == 20
        assert config.exit_sma_period == 21
        assert config.vol_window == 20
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen_model(self) -> None:
        """frozen 모델이므로 속성 수정 불가."""
        config = TtmSqueezeConfig()

        with pytest.raises(ValidationError):
            config.bb_period = 30  # type: ignore[misc]

    def test_custom_values(self) -> None:
        """커스텀 값으로 생성 가능."""
        config = TtmSqueezeConfig(
            bb_period=15,
            bb_std=1.5,
            kc_period=15,
            kc_mult=2.0,
            mom_period=10,
            exit_sma_period=10,
            vol_window=30,
            vol_target=0.30,
        )
        assert config.bb_period == 15
        assert config.bb_std == 1.5
        assert config.kc_period == 15
        assert config.kc_mult == 2.0
        assert config.mom_period == 10
        assert config.exit_sma_period == 10
        assert config.vol_window == 30
        assert config.vol_target == 0.30


class TestTtmSqueezeConfigValidation:
    """TtmSqueezeConfig 검증 테스트."""

    def test_vol_target_below_min_volatility_raises(self) -> None:
        """vol_target < min_volatility이면 ValidationError."""
        with pytest.raises(ValidationError, match="vol_target"):
            TtmSqueezeConfig(vol_target=0.05, min_volatility=0.10)

    def test_vol_target_equal_min_volatility(self) -> None:
        """vol_target == min_volatility는 허용."""
        config = TtmSqueezeConfig(vol_target=0.10, min_volatility=0.10)
        assert config.vol_target == config.min_volatility

    def test_bb_period_min(self) -> None:
        """bb_period 최소값 5 허용."""
        config = TtmSqueezeConfig(bb_period=5)
        assert config.bb_period == 5

    def test_bb_period_max(self) -> None:
        """bb_period 최대값 50 허용."""
        config = TtmSqueezeConfig(bb_period=50)
        assert config.bb_period == 50

    def test_bb_period_below_min_raises(self) -> None:
        """bb_period < 5이면 ValidationError."""
        with pytest.raises(ValidationError):
            TtmSqueezeConfig(bb_period=4)

    def test_bb_period_above_max_raises(self) -> None:
        """bb_period > 50이면 ValidationError."""
        with pytest.raises(ValidationError):
            TtmSqueezeConfig(bb_period=51)

    def test_kc_mult_range(self) -> None:
        """kc_mult 범위 검증."""
        config_min = TtmSqueezeConfig(kc_mult=0.5)
        assert config_min.kc_mult == 0.5

        config_max = TtmSqueezeConfig(kc_mult=3.0)
        assert config_max.kc_mult == 3.0

        with pytest.raises(ValidationError):
            TtmSqueezeConfig(kc_mult=0.4)

        with pytest.raises(ValidationError):
            TtmSqueezeConfig(kc_mult=3.1)


class TestWarmupPeriods:
    """warmup_periods 테스트."""

    def test_warmup_default(self) -> None:
        """기본값에서 warmup = max(20, 20, 20, 21, 20) + 1 = 22."""
        config = TtmSqueezeConfig()
        assert config.warmup_periods() == 22

    def test_warmup_custom(self) -> None:
        """커스텀 파라미터에서 warmup 계산."""
        config = TtmSqueezeConfig(
            bb_period=30,
            kc_period=10,
            mom_period=10,
            exit_sma_period=10,
            vol_window=10,
        )
        # max(30, 10, 10, 10, 10) + 1 = 31
        assert config.warmup_periods() == 31

    def test_warmup_exit_sma_dominant(self) -> None:
        """exit_sma_period가 가장 길 때."""
        config = TtmSqueezeConfig(
            bb_period=10,
            kc_period=10,
            mom_period=10,
            exit_sma_period=50,
            vol_window=10,
        )
        assert config.warmup_periods() == 51

    def test_warmup_vol_window_dominant(self) -> None:
        """vol_window가 가장 길 때."""
        config = TtmSqueezeConfig(
            bb_period=10,
            kc_period=10,
            mom_period=10,
            exit_sma_period=10,
            vol_window=100,
        )
        assert config.warmup_periods() == 101


class TestForTimeframe:
    """for_timeframe 팩토리 테스트."""

    def test_for_timeframe_1d(self) -> None:
        """for_timeframe('1d')은 annualization=365.0."""
        config = TtmSqueezeConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_4h(self) -> None:
        """for_timeframe('4h')은 annualization=2190.0."""
        config = TtmSqueezeConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_1h(self) -> None:
        """for_timeframe('1h')은 annualization=8760.0."""
        config = TtmSqueezeConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_unknown(self) -> None:
        """알 수 없는 타임프레임은 기본값 365.0 사용."""
        config = TtmSqueezeConfig.for_timeframe("7h")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe에 kwargs 오버라이드."""
        config = TtmSqueezeConfig.for_timeframe("4h", vol_target=0.30)
        assert config.annualization_factor == 2190.0
        assert config.vol_target == 0.30


class TestPresets:
    """conservative/aggressive 프리셋 테스트."""

    def test_conservative_preset(self) -> None:
        """conservative() 프리셋 검증."""
        config = TtmSqueezeConfig.conservative()

        assert config.bb_std == 2.0
        assert config.kc_mult == 2.0
        assert config.vol_target == 0.30
        assert config.min_volatility == 0.08

    def test_aggressive_preset(self) -> None:
        """aggressive() 프리셋 검증."""
        config = TtmSqueezeConfig.aggressive()

        assert config.bb_std == 1.5
        assert config.kc_mult == 1.0
        assert config.vol_target == 0.50
        assert config.min_volatility == 0.05


class TestShortModeConfig:
    """ShortMode 설정 테스트."""

    def test_default_is_disabled(self) -> None:
        """기본 short_mode는 DISABLED."""
        config = TtmSqueezeConfig()
        assert config.short_mode == ShortMode.DISABLED

    def test_full_mode(self) -> None:
        """FULL 모드 설정 가능."""
        config = TtmSqueezeConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

    def test_short_mode_from_int(self) -> None:
        """정수값으로 ShortMode 설정."""
        config = TtmSqueezeConfig(short_mode=0)
        assert config.short_mode == ShortMode.DISABLED

        config_full = TtmSqueezeConfig(short_mode=2)
        assert config_full.short_mode == ShortMode.FULL
