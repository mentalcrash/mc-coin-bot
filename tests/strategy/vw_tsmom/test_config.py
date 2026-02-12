"""Tests for VWTSMOMConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.tsmom.config import ShortMode
from src.strategy.vw_tsmom.config import VWTSMOMConfig


class TestShortModeEnum:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode를 올바르게 수용."""
        config = VWTSMOMConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = VWTSMOMConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestVWTSMOMConfig:
    """VWTSMOMConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = VWTSMOMConfig()

        assert config.lookback == 21
        assert config.vol_window == 30
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = VWTSMOMConfig()

        with pytest.raises(ValidationError):
            config.lookback = 30  # type: ignore[misc]

    def test_lookback_range(self):
        """lookback 범위 검증."""
        config = VWTSMOMConfig(lookback=5)
        assert config.lookback == 5

        config = VWTSMOMConfig(lookback=120)
        assert config.lookback == 120

        with pytest.raises(ValidationError):
            VWTSMOMConfig(lookback=4)

        with pytest.raises(ValidationError):
            VWTSMOMConfig(lookback=121)

    def test_vol_target_range(self):
        """vol_target 범위 검증."""
        config = VWTSMOMConfig(vol_target=0.05)
        assert config.vol_target == 0.05

        config = VWTSMOMConfig(vol_target=1.0)
        assert config.vol_target == 1.0

        with pytest.raises(ValidationError):
            VWTSMOMConfig(vol_target=0.04)

        with pytest.raises(ValidationError):
            VWTSMOMConfig(vol_target=1.1)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        # 유효한 경우
        config = VWTSMOMConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # 경계값 (같은 값)
        config = VWTSMOMConfig(vol_target=0.10, min_volatility=0.10)
        assert config.vol_target >= config.min_volatility

        # vol_target < min_volatility는 에러
        with pytest.raises(ValidationError):
            VWTSMOMConfig(vol_target=0.05, min_volatility=0.10)

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = VWTSMOMConfig(lookback=21, vol_window=30)
        # max(21, 30) + 1 = 31
        assert config.warmup_periods() == 31

    def test_warmup_periods_custom(self):
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = VWTSMOMConfig(lookback=50, vol_window=30)
        # max(50, 30) + 1 = 51
        assert config.warmup_periods() == 51

    def test_for_timeframe(self):
        """for_timeframe() 타임프레임별 설정 생성."""
        config_1d = VWTSMOMConfig.for_timeframe("1d")
        assert config_1d.annualization_factor == 365.0
        assert config_1d.lookback == 7

        config_1h = VWTSMOMConfig.for_timeframe("1h")
        assert config_1h.annualization_factor == 8760.0
        assert config_1h.lookback == 24

        config_4h = VWTSMOMConfig.for_timeframe("4h")
        assert config_4h.annualization_factor == 2190.0
        assert config_4h.lookback == 24

        # 알 수 없는 타임프레임은 기본값
        config_unknown = VWTSMOMConfig.for_timeframe("7h")
        assert config_unknown.annualization_factor == 8760.0
        assert config_unknown.lookback == 24

    def test_conservative(self):
        """conservative() 보수적 설정."""
        config = VWTSMOMConfig.conservative()
        assert config.lookback == 48
        assert config.vol_window == 48
        assert config.vol_target == 0.15
        assert config.min_volatility == 0.08

    def test_aggressive(self):
        """aggressive() 공격적 설정."""
        config = VWTSMOMConfig.aggressive()
        assert config.lookback == 10
        assert config.vol_window == 10
        assert config.vol_target == 0.50
        assert config.min_volatility == 0.05
