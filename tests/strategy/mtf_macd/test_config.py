"""Tests for MtfMacdConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.mtf_macd.config import MtfMacdConfig, ShortMode


class TestMtfMacdConfigDefaults:
    """MtfMacdConfig 기본값 테스트."""

    def test_default_values(self) -> None:
        """기본값이 올바르게 설정되는지 확인."""
        config = MtfMacdConfig()

        assert config.fast_period == 12
        assert config.slow_period == 26
        assert config.signal_period == 9
        assert config.vol_window == 20
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen_model(self) -> None:
        """frozen 모델이므로 속성 수정 불가."""
        config = MtfMacdConfig()

        with pytest.raises(ValidationError):
            config.fast_period = 20  # type: ignore[misc]


class TestMtfMacdConfigValidation:
    """MtfMacdConfig 검증 테스트."""

    def test_fast_less_than_slow_valid(self) -> None:
        """fast_period < slow_period (valid)."""
        config = MtfMacdConfig(fast_period=10, slow_period=20)
        assert config.fast_period < config.slow_period

    def test_fast_equal_slow_raises(self) -> None:
        """fast_period == slow_period이면 ValidationError."""
        with pytest.raises(ValidationError, match="fast_period"):
            MtfMacdConfig(fast_period=20, slow_period=20)

    def test_fast_greater_than_slow_raises(self) -> None:
        """fast_period > slow_period이면 ValidationError."""
        with pytest.raises(ValidationError, match="fast_period"):
            MtfMacdConfig(fast_period=30, slow_period=20)

    def test_vol_target_less_than_min_raises(self) -> None:
        """vol_target < min_volatility이면 ValidationError."""
        with pytest.raises(ValidationError, match="vol_target"):
            MtfMacdConfig(vol_target=0.03, min_volatility=0.10)

    def test_vol_target_equal_min_volatility(self) -> None:
        """vol_target == min_volatility는 허용."""
        config = MtfMacdConfig(vol_target=0.10, min_volatility=0.10)
        assert config.vol_target == config.min_volatility


class TestMtfMacdConfigWarmup:
    """워밍업 기간 테스트."""

    def test_warmup_periods_default(self) -> None:
        """기본 설정의 warmup = 26 + 9 + 1 = 36."""
        config = MtfMacdConfig()
        assert config.warmup_periods() == 26 + 9 + 1

    def test_warmup_periods_custom(self) -> None:
        """커스텀 파라미터에서도 warmup 정확."""
        config = MtfMacdConfig(fast_period=8, slow_period=17, signal_period=5)
        assert config.warmup_periods() == 17 + 5 + 1


class TestMtfMacdConfigTimeframe:
    """타임프레임별 설정 테스트."""

    def test_for_timeframe_1d(self) -> None:
        """for_timeframe('1d')은 annualization=365.0."""
        config = MtfMacdConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_1h(self) -> None:
        """for_timeframe('1h')은 annualization=8760.0."""
        config = MtfMacdConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_4h(self) -> None:
        """for_timeframe('4h')은 annualization=2190.0."""
        config = MtfMacdConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe에 kwargs 오버라이드."""
        config = MtfMacdConfig.for_timeframe("1d", vol_target=0.30)
        assert config.annualization_factor == 365.0
        assert config.vol_target == 0.30

    def test_for_timeframe_unknown(self) -> None:
        """알 수 없는 타임프레임은 기본값 365.0 사용."""
        config = MtfMacdConfig.for_timeframe("3h")
        assert config.annualization_factor == 365.0


class TestMtfMacdConfigPresets:
    """프리셋 테스트."""

    def test_conservative_preset(self) -> None:
        """conservative() 프리셋 검증."""
        config = MtfMacdConfig.conservative()

        assert config.fast_period == 12
        assert config.slow_period == 26
        assert config.signal_period == 9
        assert config.vol_target == 0.30

    def test_aggressive_preset(self) -> None:
        """aggressive() 프리셋 검증."""
        config = MtfMacdConfig.aggressive()

        assert config.fast_period == 8
        assert config.slow_period == 17
        assert config.signal_period == 9
        assert config.vol_target == 0.50


class TestMtfMacdConfigFieldRanges:
    """필드 범위 검증 테스트."""

    def test_fast_period_min(self) -> None:
        """fast_period 최소값 5 허용."""
        config = MtfMacdConfig(fast_period=5, slow_period=26)
        assert config.fast_period == 5

    def test_fast_period_max(self) -> None:
        """fast_period 최대값 50 허용."""
        config = MtfMacdConfig(fast_period=50, slow_period=100)
        assert config.fast_period == 50

    def test_fast_period_below_min(self) -> None:
        """fast_period < 5이면 ValidationError."""
        with pytest.raises(ValidationError):
            MtfMacdConfig(fast_period=4)

    def test_fast_period_above_max(self) -> None:
        """fast_period > 50이면 ValidationError."""
        with pytest.raises(ValidationError):
            MtfMacdConfig(fast_period=51)

    def test_slow_period_min(self) -> None:
        """slow_period 최소값 10 허용."""
        config = MtfMacdConfig(fast_period=5, slow_period=10)
        assert config.slow_period == 10

    def test_slow_period_below_min(self) -> None:
        """slow_period < 10이면 ValidationError."""
        with pytest.raises(ValidationError):
            MtfMacdConfig(slow_period=9)

    def test_signal_period_min(self) -> None:
        """signal_period 최소값 3 허용."""
        config = MtfMacdConfig(signal_period=3)
        assert config.signal_period == 3

    def test_signal_period_above_max(self) -> None:
        """signal_period > 30이면 ValidationError."""
        with pytest.raises(ValidationError):
            MtfMacdConfig(signal_period=31)


class TestShortModeConfig:
    """ShortMode 설정 테스트."""

    def test_default_is_disabled(self) -> None:
        """기본 short_mode는 DISABLED."""
        config = MtfMacdConfig()
        assert config.short_mode == ShortMode.DISABLED

    def test_full_mode(self) -> None:
        """FULL 모드 설정 가능."""
        config = MtfMacdConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

    def test_short_mode_values(self) -> None:
        """ShortMode enum 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.FULL == 2
