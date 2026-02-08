"""Tests for KAMAConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.kama.config import KAMAConfig, ShortMode


class TestKAMAConfig:
    """KAMAConfig 테스트."""

    def test_default_values(self):
        """기본값이 올바르게 설정되는지 확인."""
        config = KAMAConfig()

        assert config.er_lookback == 10
        assert config.fast_period == 2
        assert config.slow_period == 30
        assert config.atr_period == 14
        assert config.atr_multiplier == 1.5
        assert config.vol_window == 30
        assert config.vol_target == 0.30
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        """frozen 모델이므로 속성 수정 불가."""
        config = KAMAConfig()

        with pytest.raises(ValidationError):
            config.er_lookback = 20  # type: ignore[misc]

    def test_er_lookback_range(self):
        """er_lookback 범위 검증 (5-100)."""
        KAMAConfig(er_lookback=5)
        KAMAConfig(er_lookback=100)

        with pytest.raises(ValidationError):
            KAMAConfig(er_lookback=4)

        with pytest.raises(ValidationError):
            KAMAConfig(er_lookback=101)

    def test_fast_period_range(self):
        """fast_period 범위 검증 (2-10)."""
        KAMAConfig(fast_period=2)
        KAMAConfig(fast_period=10)

        with pytest.raises(ValidationError):
            KAMAConfig(fast_period=1)

        with pytest.raises(ValidationError):
            KAMAConfig(fast_period=11)

    def test_slow_period_range(self):
        """slow_period 범위 검증 (10-100)."""
        KAMAConfig(slow_period=10, fast_period=2)
        KAMAConfig(slow_period=100)

        with pytest.raises(ValidationError):
            KAMAConfig(slow_period=9)

        with pytest.raises(ValidationError):
            KAMAConfig(slow_period=101)

    def test_atr_multiplier_range(self):
        """atr_multiplier 범위 검증 (0.5-5.0)."""
        KAMAConfig(atr_multiplier=0.5)
        KAMAConfig(atr_multiplier=5.0)

        with pytest.raises(ValidationError):
            KAMAConfig(atr_multiplier=0.4)

        with pytest.raises(ValidationError):
            KAMAConfig(atr_multiplier=5.1)

    def test_slow_greater_than_fast_validation(self):
        """slow_period > fast_period 검증."""
        KAMAConfig(fast_period=2, slow_period=30)
        KAMAConfig(fast_period=5, slow_period=10)

        with pytest.raises(ValidationError):
            KAMAConfig(fast_period=10, slow_period=10)

        with pytest.raises(ValidationError):
            KAMAConfig(fast_period=10, slow_period=5)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        KAMAConfig(vol_target=0.20, min_volatility=0.05)
        KAMAConfig(vol_target=0.05, min_volatility=0.05)

        with pytest.raises(ValidationError):
            KAMAConfig(vol_target=0.04, min_volatility=0.05)

    def test_warmup_periods(self):
        """warmup_periods가 최대 기간 + 1을 반환."""
        config = KAMAConfig(er_lookback=10, slow_period=30, vol_window=20, atr_period=14)
        # max(10, 30, 20, 14) + 1 = 31
        assert config.warmup_periods() == 31

    def test_warmup_periods_vol_window_dominant(self):
        """vol_window가 가장 클 때 warmup 검증."""
        config = KAMAConfig(er_lookback=5, slow_period=10, vol_window=50, atr_period=10)
        # max(5, 10, 50, 10) + 1 = 51
        assert config.warmup_periods() == 51

    def test_warmup_periods_er_lookback_dominant(self):
        """er_lookback이 가장 클 때 warmup 검증."""
        config = KAMAConfig(er_lookback=80, slow_period=30, vol_window=30, atr_period=14)
        # max(80, 30, 30, 14) + 1 = 81
        assert config.warmup_periods() == 81


class TestShortMode:
    """ShortMode enum 테스트."""

    def test_values(self):
        """ShortMode 값 검증."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode 사용 가능."""
        config = KAMAConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = KAMAConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED

        config = KAMAConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY
