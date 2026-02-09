"""Tests for VolAdaptiveConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.vol_adaptive.config import ShortMode, VolAdaptiveConfig


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self) -> None:
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self) -> None:
        """Config에 ShortMode 설정 가능."""
        config = VolAdaptiveConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = VolAdaptiveConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY


class TestVolAdaptiveConfig:
    """VolAdaptiveConfig 테스트."""

    def test_default_values(self) -> None:
        """기본값으로 생성 테스트."""
        config = VolAdaptiveConfig()

        assert config.ema_fast == 10
        assert config.ema_slow == 50
        assert config.rsi_period == 14
        assert config.rsi_upper == 50.0
        assert config.rsi_lower == 50.0
        assert config.adx_period == 14
        assert config.adx_threshold == 20.0
        assert config.vol_window == 20
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.atr_period == 14
        assert config.use_log_returns is True
        assert config.short_mode == ShortMode.DISABLED
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self) -> None:
        """Frozen 모델이므로 변경 불가."""
        config = VolAdaptiveConfig()

        with pytest.raises(ValidationError):
            config.ema_fast = 20  # type: ignore[misc]

    def test_ema_fast_range(self) -> None:
        """ema_fast 범위 검증."""
        config = VolAdaptiveConfig(ema_fast=3)
        assert config.ema_fast == 3

        config = VolAdaptiveConfig(ema_fast=30, ema_slow=50)
        assert config.ema_fast == 30

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(ema_fast=2)

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(ema_fast=31)

    def test_ema_slow_range(self) -> None:
        """ema_slow 범위 검증."""
        config = VolAdaptiveConfig(ema_slow=20, ema_fast=10)
        assert config.ema_slow == 20

        config = VolAdaptiveConfig(ema_slow=200)
        assert config.ema_slow == 200

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(ema_slow=19)

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(ema_slow=201)

    def test_rsi_period_range(self) -> None:
        """rsi_period 범위 검증."""
        config = VolAdaptiveConfig(rsi_period=5)
        assert config.rsi_period == 5

        config = VolAdaptiveConfig(rsi_period=30)
        assert config.rsi_period == 30

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(rsi_period=4)

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(rsi_period=31)

    def test_adx_period_range(self) -> None:
        """adx_period 범위 검증."""
        config = VolAdaptiveConfig(adx_period=5)
        assert config.adx_period == 5

        config = VolAdaptiveConfig(adx_period=30)
        assert config.adx_period == 30

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(adx_period=4)

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(adx_period=31)

    def test_adx_threshold_range(self) -> None:
        """adx_threshold 범위 검증."""
        config = VolAdaptiveConfig(adx_threshold=10.0)
        assert config.adx_threshold == 10.0

        config = VolAdaptiveConfig(adx_threshold=40.0)
        assert config.adx_threshold == 40.0

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(adx_threshold=9.9)

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(adx_threshold=40.1)

    def test_slow_greater_than_fast_validation(self) -> None:
        """ema_slow > ema_fast 검증."""
        # 유효한 경우
        config = VolAdaptiveConfig(ema_fast=10, ema_slow=50)
        assert config.ema_slow > config.ema_fast

        # ema_slow <= ema_fast는 에러
        with pytest.raises(ValidationError):
            VolAdaptiveConfig(ema_fast=20, ema_slow=20)

        with pytest.raises(ValidationError):
            VolAdaptiveConfig(ema_fast=30, ema_slow=25)

    def test_vol_target_gte_min_volatility(self) -> None:
        """vol_target >= min_volatility 검증."""
        # 유효한 경우
        config = VolAdaptiveConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # vol_target < min_volatility는 에러
        with pytest.raises(ValidationError):
            VolAdaptiveConfig(vol_target=0.05, min_volatility=0.10)

    def test_warmup_periods(self) -> None:
        """warmup_periods() 테스트."""
        config = VolAdaptiveConfig(
            ema_fast=10,
            ema_slow=50,
            vol_window=20,
            adx_period=14,
            rsi_period=14,
            atr_period=14,
        )
        # max(50, 20, 14, 14, 14) + 1 = 51
        assert config.warmup_periods() == 51

    def test_warmup_periods_custom(self) -> None:
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = VolAdaptiveConfig(
            ema_fast=10,
            ema_slow=100,
            vol_window=60,
            adx_period=30,
            rsi_period=30,
            atr_period=50,
        )
        # max(100, 60, 30, 30, 50) + 1 = 101
        assert config.warmup_periods() == 101
