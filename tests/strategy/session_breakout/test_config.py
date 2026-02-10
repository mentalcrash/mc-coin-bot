"""Tests for SessionBreakoutConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.session_breakout.config import SessionBreakoutConfig, ShortMode


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode를 올바르게 수용."""
        config = SessionBreakoutConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY


class TestSessionBreakoutConfig:
    """SessionBreakoutConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = SessionBreakoutConfig()

        assert config.asian_start_hour == 0
        assert config.asian_end_hour == 8
        assert config.trade_end_hour == 20
        assert config.exit_hour == 22
        assert config.range_pctl_window == 720
        assert config.range_pctl_threshold == 50.0
        assert config.tp_multiplier == 1.5
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 8760.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = SessionBreakoutConfig()

        with pytest.raises(ValidationError):
            config.asian_start_hour = 2  # type: ignore[misc]

    def test_asian_hour_range(self):
        """asian session 시간 검증."""
        config = SessionBreakoutConfig(asian_start_hour=2, asian_end_hour=10)
        assert config.asian_start_hour == 2
        assert config.asian_end_hour == 10

    def test_asian_end_must_be_after_start(self):
        """asian_end_hour > asian_start_hour 검증."""
        with pytest.raises(ValidationError):
            SessionBreakoutConfig(asian_start_hour=8, asian_end_hour=8)

        with pytest.raises(ValidationError):
            SessionBreakoutConfig(asian_start_hour=10, asian_end_hour=5)

    def test_exit_must_be_after_trade_end(self):
        """exit_hour > trade_end_hour 검증."""
        with pytest.raises(ValidationError):
            SessionBreakoutConfig(trade_end_hour=22, exit_hour=22)

    def test_range_pctl_window_range(self):
        """range_pctl_window 범위 검증."""
        config = SessionBreakoutConfig(range_pctl_window=48)
        assert config.range_pctl_window == 48

        with pytest.raises(ValidationError):
            SessionBreakoutConfig(range_pctl_window=47)

        with pytest.raises(ValidationError):
            SessionBreakoutConfig(range_pctl_window=2161)

    def test_range_pctl_threshold_range(self):
        """range_pctl_threshold 범위 검증."""
        config = SessionBreakoutConfig(range_pctl_threshold=10.0)
        assert config.range_pctl_threshold == 10.0

        with pytest.raises(ValidationError):
            SessionBreakoutConfig(range_pctl_threshold=9.0)

        with pytest.raises(ValidationError):
            SessionBreakoutConfig(range_pctl_threshold=91.0)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        with pytest.raises(ValidationError):
            SessionBreakoutConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = SessionBreakoutConfig(range_pctl_window=720)
        assert config.warmup_periods() == 721

    def test_warmup_periods_custom(self):
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = SessionBreakoutConfig(range_pctl_window=1440)
        assert config.warmup_periods() == 1441
