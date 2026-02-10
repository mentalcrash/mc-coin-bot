"""Tests for RangeSqueezeConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.range_squeeze.config import RangeSqueezeConfig, ShortMode


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode를 올바르게 수용."""
        config = RangeSqueezeConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = RangeSqueezeConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestRangeSqueezeConfig:
    """RangeSqueezeConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = RangeSqueezeConfig()

        assert config.nr_period == 7
        assert config.lookback == 20
        assert config.squeeze_threshold == 0.5
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = RangeSqueezeConfig()

        with pytest.raises(ValidationError):
            config.nr_period = 10  # type: ignore[misc]

    def test_nr_period_range(self):
        """nr_period 범위 검증."""
        config = RangeSqueezeConfig(nr_period=3)
        assert config.nr_period == 3

        config = RangeSqueezeConfig(nr_period=20)
        assert config.nr_period == 20

        with pytest.raises(ValidationError):
            RangeSqueezeConfig(nr_period=2)

        with pytest.raises(ValidationError):
            RangeSqueezeConfig(nr_period=21)

    def test_lookback_range(self):
        """lookback 범위 검증."""
        config = RangeSqueezeConfig(lookback=10)
        assert config.lookback == 10

        config = RangeSqueezeConfig(lookback=60)
        assert config.lookback == 60

        with pytest.raises(ValidationError):
            RangeSqueezeConfig(lookback=9)

        with pytest.raises(ValidationError):
            RangeSqueezeConfig(lookback=61)

    def test_squeeze_threshold_range(self):
        """squeeze_threshold 범위 검증."""
        config = RangeSqueezeConfig(squeeze_threshold=0.2)
        assert config.squeeze_threshold == 0.2

        config = RangeSqueezeConfig(squeeze_threshold=0.9)
        assert config.squeeze_threshold == 0.9

        with pytest.raises(ValidationError):
            RangeSqueezeConfig(squeeze_threshold=0.1)

        with pytest.raises(ValidationError):
            RangeSqueezeConfig(squeeze_threshold=1.0)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        config = RangeSqueezeConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        with pytest.raises(ValidationError):
            RangeSqueezeConfig(vol_target=0.03, min_volatility=0.05)

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = RangeSqueezeConfig(
            nr_period=7,
            lookback=20,
            atr_period=14,
        )
        # max(20, 7, 14) + 1 = 21
        assert config.warmup_periods() == 21

    def test_warmup_periods_custom(self):
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = RangeSqueezeConfig(
            nr_period=15,
            lookback=40,
            atr_period=50,
        )
        # max(40, 15, 50) + 1 = 51
        assert config.warmup_periods() == 51
