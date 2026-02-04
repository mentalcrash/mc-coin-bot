"""Tests for AdaptiveBreakoutConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.breakout.config import AdaptiveBreakoutConfig


class TestAdaptiveBreakoutConfig:
    """AdaptiveBreakoutConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = AdaptiveBreakoutConfig()

        assert config.channel_period == 20
        assert config.atr_period == 14
        assert config.k_value == 1.0  # 암호화폐에 최적화된 기본값
        assert config.vol_target == 0.40
        assert config.long_only is False
        assert config.adaptive_threshold is True

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = AdaptiveBreakoutConfig()

        with pytest.raises(ValidationError):
            config.channel_period = 30  # type: ignore[misc]

    def test_validation_channel_period(self):
        """channel_period 범위 검증."""
        # 유효한 범위
        config = AdaptiveBreakoutConfig(channel_period=5)
        assert config.channel_period == 5

        config = AdaptiveBreakoutConfig(channel_period=100)
        assert config.channel_period == 100

        # 범위 초과
        with pytest.raises(ValidationError):
            AdaptiveBreakoutConfig(channel_period=4)

        with pytest.raises(ValidationError):
            AdaptiveBreakoutConfig(channel_period=101)

    def test_validation_k_value(self):
        """k_value 범위 검증."""
        config = AdaptiveBreakoutConfig(k_value=0.5)
        assert config.k_value == 0.5

        config = AdaptiveBreakoutConfig(k_value=5.0)
        assert config.k_value == 5.0

        with pytest.raises(ValidationError):
            AdaptiveBreakoutConfig(k_value=0.4)

        with pytest.raises(ValidationError):
            AdaptiveBreakoutConfig(k_value=5.1)

    def test_validation_vol_target_vs_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        # 유효한 경우
        config = AdaptiveBreakoutConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # vol_target < min_volatility는 에러
        with pytest.raises(ValidationError):
            AdaptiveBreakoutConfig(vol_target=0.04, min_volatility=0.05)

    def test_validation_k_value_without_adaptive(self):
        """k_value < 1.0 + adaptive_threshold=False는 경고."""
        # adaptive_threshold=True면 OK
        config = AdaptiveBreakoutConfig(k_value=0.8, adaptive_threshold=True)
        assert config.k_value == 0.8

        # adaptive_threshold=False면 에러
        with pytest.raises(ValidationError):
            AdaptiveBreakoutConfig(k_value=0.8, adaptive_threshold=False)

    def test_factory_conservative(self):
        """conservative() 팩토리 메서드 테스트."""
        config = AdaptiveBreakoutConfig.conservative()

        assert config.channel_period == 30
        assert config.k_value == 2.0
        assert config.atr_period == 20
        assert config.use_trailing_stop is True

    def test_factory_aggressive(self):
        """aggressive() 팩토리 메서드 테스트."""
        config = AdaptiveBreakoutConfig.aggressive()

        assert config.channel_period == 10
        assert config.k_value == 1.0
        assert config.atr_period == 10

    def test_factory_for_timeframe(self):
        """for_timeframe() 팩토리 메서드 테스트."""
        # 일봉
        config_1d = AdaptiveBreakoutConfig.for_timeframe("1d")
        assert config_1d.channel_period == 20
        assert config_1d.annualization_factor == 365.0

        # 시간봉
        config_1h = AdaptiveBreakoutConfig.for_timeframe("1h")
        assert config_1h.channel_period == 24
        assert config_1h.annualization_factor == 8760.0

        # 4시간봉
        config_4h = AdaptiveBreakoutConfig.for_timeframe("4h")
        assert config_4h.channel_period == 30
        assert config_4h.annualization_factor == 2190.0

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = AdaptiveBreakoutConfig(
            channel_period=20,
            atr_period=14,
            volatility_lookback=25,
        )

        # max(20, 14, 25) + 1 = 26
        assert config.warmup_periods() == 26
