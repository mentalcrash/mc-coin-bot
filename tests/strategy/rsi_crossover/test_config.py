"""Tests for RSICrossoverConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.rsi_crossover.config import RSICrossoverConfig
from src.strategy.tsmom.config import ShortMode


class TestRSICrossoverConfig:
    """RSICrossoverConfig 기본 테스트."""

    def test_default_values(self) -> None:
        """기본값이 올바르게 설정되는지 확인."""
        config = RSICrossoverConfig()

        assert config.rsi_period == 14
        assert config.entry_oversold == 30.0
        assert config.entry_overbought == 70.0
        assert config.exit_long == 60.0
        assert config.exit_short == 40.0
        assert config.annualization_factor == 2190.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen_model(self) -> None:
        """frozen 모델이므로 속성 수정 불가."""
        config = RSICrossoverConfig()

        with pytest.raises(ValidationError):
            config.rsi_period = 20  # type: ignore[misc]

    def test_threshold_ordering_valid(self) -> None:
        """entry_oversold < exit_short < exit_long < entry_overbought (valid)."""
        config = RSICrossoverConfig(
            entry_oversold=25.0,
            exit_short=40.0,
            exit_long=60.0,
            entry_overbought=75.0,
        )
        assert config.entry_oversold < config.exit_short
        assert config.exit_short < config.exit_long
        assert config.exit_long < config.entry_overbought

    def test_threshold_ordering_invalid(self) -> None:
        """Threshold 순서가 역전되면 ValidationError."""
        # exit_short > exit_long (역전, but within field bounds)
        with pytest.raises(ValidationError, match="Threshold order must be"):
            RSICrossoverConfig(
                entry_oversold=25.0,
                exit_short=52.0,
                exit_long=48.0,
                entry_overbought=75.0,
            )

    def test_threshold_ordering_equal_raises(self) -> None:
        """Threshold 값이 같으면 ValidationError."""
        with pytest.raises(ValidationError):
            RSICrossoverConfig(
                entry_oversold=40.0,
                exit_short=40.0,
                exit_long=60.0,
                entry_overbought=70.0,
            )

    def test_vol_target_validation(self) -> None:
        """vol_target < min_volatility이면 ValidationError."""
        with pytest.raises(ValidationError, match="vol_target"):
            RSICrossoverConfig(vol_target=0.05, min_volatility=0.10)

    def test_vol_target_equal_min_volatility(self) -> None:
        """vol_target == min_volatility는 허용."""
        config = RSICrossoverConfig(vol_target=0.10, min_volatility=0.10)
        assert config.vol_target == config.min_volatility

    def test_for_timeframe_4h(self) -> None:
        """for_timeframe('4h')은 기본 annualization=2190.0."""
        config = RSICrossoverConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_1d(self) -> None:
        """for_timeframe('1d')은 annualization=365.0."""
        config = RSICrossoverConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_1h(self) -> None:
        """for_timeframe('1h')은 annualization=8760.0."""
        config = RSICrossoverConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe에 kwargs 오버라이드."""
        config = RSICrossoverConfig.for_timeframe("4h", vol_target=0.30)
        assert config.annualization_factor == 2190.0
        assert config.vol_target == 0.30

    def test_for_timeframe_unknown(self) -> None:
        """알 수 없는 타임프레임은 기본값 2190.0 사용."""
        config = RSICrossoverConfig.for_timeframe("3h")
        assert config.annualization_factor == 2190.0

    def test_warmup_periods(self) -> None:
        """warmup_periods = rsi_period + vol_window + 2."""
        config = RSICrossoverConfig(rsi_period=14, vol_window=30)
        assert config.warmup_periods() == 14 + 30 + 2

    def test_warmup_periods_custom(self) -> None:
        """커스텀 파라미터에서도 warmup_periods 정확."""
        config = RSICrossoverConfig(rsi_period=10, vol_window=20)
        assert config.warmup_periods() == 10 + 20 + 2

    def test_conservative_preset(self) -> None:
        """conservative() 프리셋 검증."""
        config = RSICrossoverConfig.conservative()

        assert config.rsi_period == 14
        assert config.entry_oversold == 25.0
        assert config.entry_overbought == 75.0
        assert config.exit_long == 55.0
        assert config.exit_short == 45.0
        assert config.vol_target == 0.15
        assert config.min_volatility == 0.08

    def test_aggressive_preset(self) -> None:
        """aggressive() 프리셋 검증."""
        config = RSICrossoverConfig.aggressive()

        assert config.rsi_period == 10
        assert config.entry_oversold == 35.0
        assert config.entry_overbought == 65.0
        assert config.exit_long == 55.0
        assert config.exit_short == 45.0
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05


class TestRSIPeriodRange:
    """RSI period 범위 검증."""

    def test_rsi_period_min(self) -> None:
        """rsi_period 최소값 5 허용."""
        config = RSICrossoverConfig(rsi_period=5)
        assert config.rsi_period == 5

    def test_rsi_period_max(self) -> None:
        """rsi_period 최대값 30 허용."""
        config = RSICrossoverConfig(rsi_period=30)
        assert config.rsi_period == 30

    def test_rsi_period_below_min(self) -> None:
        """rsi_period < 5이면 ValidationError."""
        with pytest.raises(ValidationError):
            RSICrossoverConfig(rsi_period=4)

    def test_rsi_period_above_max(self) -> None:
        """rsi_period > 30이면 ValidationError."""
        with pytest.raises(ValidationError):
            RSICrossoverConfig(rsi_period=31)


class TestShortModeConfig:
    """ShortMode 설정 테스트."""

    def test_default_is_full(self) -> None:
        """기본 short_mode는 FULL."""
        config = RSICrossoverConfig()
        assert config.short_mode == ShortMode.FULL

    def test_disabled_mode(self) -> None:
        """DISABLED 모드 설정 가능."""
        config = RSICrossoverConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED

    def test_hedge_only_mode(self) -> None:
        """HEDGE_ONLY 모드 설정 가능."""
        config = RSICrossoverConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY
