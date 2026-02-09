"""Unit tests for XSMOMConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.tsmom.config import ShortMode
from src.strategy.xsmom.config import XSMOMConfig


class TestConfigDefaults:
    """XSMOMConfig 기본값 테스트."""

    def test_default_lookback(self) -> None:
        """기본 lookback이 21인지 확인."""
        config = XSMOMConfig()
        assert config.lookback == 21

    def test_default_holding_period(self) -> None:
        """기본 holding_period가 7인지 확인."""
        config = XSMOMConfig()
        assert config.holding_period == 7

    def test_default_vol_window(self) -> None:
        """기본 vol_window가 30인지 확인."""
        config = XSMOMConfig()
        assert config.vol_window == 30

    def test_default_vol_target(self) -> None:
        """기본 vol_target이 0.35인지 확인."""
        config = XSMOMConfig()
        assert config.vol_target == 0.35

    def test_default_min_volatility(self) -> None:
        """기본 min_volatility가 0.05인지 확인."""
        config = XSMOMConfig()
        assert config.min_volatility == 0.05

    def test_default_annualization_factor(self) -> None:
        """기본 annualization_factor가 365.0인지 확인."""
        config = XSMOMConfig()
        assert config.annualization_factor == 365.0

    def test_default_use_log_returns(self) -> None:
        """기본 use_log_returns가 True인지 확인."""
        config = XSMOMConfig()
        assert config.use_log_returns is True

    def test_default_short_mode(self) -> None:
        """기본 short_mode가 FULL인지 확인."""
        config = XSMOMConfig()
        assert config.short_mode == ShortMode.FULL

    def test_frozen_model(self) -> None:
        """frozen=True로 인해 속성 변경이 불가한지 확인."""
        config = XSMOMConfig()
        with pytest.raises(ValidationError):
            config.vol_target = 0.50  # type: ignore[misc]


class TestConfigValidation:
    """XSMOMConfig 검증 테스트."""

    def test_vol_target_less_than_min_volatility_raises(self) -> None:
        """vol_target < min_volatility일 때 ValidationError 발생."""
        with pytest.raises(ValidationError, match="vol_target"):
            XSMOMConfig(vol_target=0.03, min_volatility=0.05)

    def test_vol_target_equal_min_volatility(self) -> None:
        """vol_target == min_volatility일 때 통과."""
        config = XSMOMConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target == config.min_volatility

    def test_lookback_range_too_low(self) -> None:
        """lookback < 5일 때 ValidationError 발생."""
        with pytest.raises(ValidationError):
            XSMOMConfig(lookback=4)

    def test_lookback_range_too_high(self) -> None:
        """lookback > 120일 때 ValidationError 발생."""
        with pytest.raises(ValidationError):
            XSMOMConfig(lookback=121)

    def test_holding_period_range_too_low(self) -> None:
        """holding_period < 1일 때 ValidationError 발생."""
        with pytest.raises(ValidationError):
            XSMOMConfig(holding_period=0)

    def test_holding_period_range_too_high(self) -> None:
        """holding_period > 30일 때 ValidationError 발생."""
        with pytest.raises(ValidationError):
            XSMOMConfig(holding_period=31)

    def test_vol_target_range_too_low(self) -> None:
        """vol_target < 0.05일 때 ValidationError 발생."""
        with pytest.raises(ValidationError):
            XSMOMConfig(vol_target=0.01)

    def test_vol_target_range_too_high(self) -> None:
        """vol_target > 1.0일 때 ValidationError 발생."""
        with pytest.raises(ValidationError):
            XSMOMConfig(vol_target=1.5)

    def test_short_mode_values(self) -> None:
        """모든 ShortMode 값으로 생성 가능."""
        for mode in ShortMode:
            config = XSMOMConfig(short_mode=mode)
            assert config.short_mode == mode


class TestConfigWarmup:
    """warmup_periods 계산 테스트."""

    def test_warmup_default(self) -> None:
        """warmup = max(lookback=21, vol_window=30) + 1 = 31."""
        config = XSMOMConfig()
        assert config.warmup_periods() == 31

    def test_warmup_lookback_dominant(self) -> None:
        """lookback > vol_window인 경우."""
        config = XSMOMConfig(lookback=60, vol_window=30)
        assert config.warmup_periods() == 61

    def test_warmup_vol_window_dominant(self) -> None:
        """vol_window > lookback인 경우."""
        config = XSMOMConfig(lookback=10, vol_window=50)
        assert config.warmup_periods() == 51


class TestConfigTimeframe:
    """for_timeframe() 팩토리 메서드 테스트."""

    def test_for_timeframe_1d(self) -> None:
        """1d 타임프레임에서 annualization_factor=365.0."""
        config = XSMOMConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_1h(self) -> None:
        """1h 타임프레임에서 annualization_factor=8760.0."""
        config = XSMOMConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_4h(self) -> None:
        """4h 타임프레임에서 annualization_factor=2190.0."""
        config = XSMOMConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_with_override(self) -> None:
        """for_timeframe에 추가 파라미터 오버라이드."""
        config = XSMOMConfig.for_timeframe("1d", vol_target=0.50)
        assert config.annualization_factor == 365.0
        assert config.vol_target == 0.50

    def test_for_timeframe_unknown_defaults_to_365(self) -> None:
        """알 수 없는 타임프레임은 365.0으로 기본값."""
        config = XSMOMConfig.for_timeframe("3m")
        assert config.annualization_factor == 365.0


class TestConfigPresets:
    """preset 팩토리 메서드 테스트."""

    def test_conservative_preset(self) -> None:
        """보수적 설정: 긴 lookback, 낮은 vol_target."""
        config = XSMOMConfig.conservative()
        assert config.lookback == 60
        assert config.holding_period == 14
        assert config.vol_target == 0.15

    def test_aggressive_preset(self) -> None:
        """공격적 설정: 짧은 lookback, 높은 vol_target."""
        config = XSMOMConfig.aggressive()
        assert config.lookback == 10
        assert config.holding_period == 3
        assert config.vol_target == 0.50

    def test_conservative_is_valid_config(self) -> None:
        """보수적 설정이 유효한 config인지 확인."""
        config = XSMOMConfig.conservative()
        assert isinstance(config, XSMOMConfig)

    def test_aggressive_is_valid_config(self) -> None:
        """공격적 설정이 유효한 config인지 확인."""
        config = XSMOMConfig.aggressive()
        assert isinstance(config, XSMOMConfig)
