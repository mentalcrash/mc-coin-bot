"""Tests for GKBreakoutConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.gk_breakout.config import GKBreakoutConfig, ShortMode


class TestGKBreakoutConfig:
    """GKBreakoutConfig 테스트."""

    def test_default_values(self):
        """기본값이 올바르게 설정되는지 확인."""
        config = GKBreakoutConfig()

        assert config.gk_lookback == 20
        assert config.compression_threshold == 0.75
        assert config.breakout_lookback == 20
        assert config.atr_period == 14
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
        config = GKBreakoutConfig()

        with pytest.raises(ValidationError):
            config.gk_lookback = 30  # type: ignore[misc]

    def test_gk_lookback_range(self):
        """gk_lookback 범위 검증 (5-100)."""
        GKBreakoutConfig(gk_lookback=5)
        GKBreakoutConfig(gk_lookback=100)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(gk_lookback=4)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(gk_lookback=101)

    def test_compression_threshold_range(self):
        """compression_threshold 범위 검증 (0.3-1.0)."""
        GKBreakoutConfig(compression_threshold=0.3)
        GKBreakoutConfig(compression_threshold=1.0)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(compression_threshold=0.29)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(compression_threshold=1.01)

    def test_breakout_lookback_range(self):
        """breakout_lookback 범위 검증 (5-100)."""
        GKBreakoutConfig(breakout_lookback=5)
        GKBreakoutConfig(breakout_lookback=100)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(breakout_lookback=4)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(breakout_lookback=101)

    def test_atr_period_range(self):
        """atr_period 범위 검증 (5-50)."""
        GKBreakoutConfig(atr_period=5)
        GKBreakoutConfig(atr_period=50)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(atr_period=4)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(atr_period=51)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        GKBreakoutConfig(vol_target=0.30, min_volatility=0.05)
        GKBreakoutConfig(vol_target=0.05, min_volatility=0.05)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(vol_target=0.04, min_volatility=0.05)

    def test_warmup_periods_gk_dominant(self):
        """gk_lookback * 2가 가장 클 때 warmup_periods 반환값."""
        config = GKBreakoutConfig(
            gk_lookback=50, breakout_lookback=20, vol_window=30, atr_period=14
        )
        # max(50*2, 20, 30, 14) + 1 = 101
        assert config.warmup_periods() == 101

    def test_warmup_periods_breakout_dominant(self):
        """breakout_lookback가 가장 클 때 warmup_periods 반환값."""
        config = GKBreakoutConfig(
            gk_lookback=5, breakout_lookback=100, vol_window=30, atr_period=14
        )
        # max(5*2, 100, 30, 14) + 1 = 101
        assert config.warmup_periods() == 101

    def test_warmup_periods_vol_window_dominant(self):
        """vol_window가 가장 클 때 warmup_periods 반환값."""
        config = GKBreakoutConfig(
            gk_lookback=5, breakout_lookback=10, vol_window=365, atr_period=14
        )
        # max(5*2, 10, 365, 14) + 1 = 366
        assert config.warmup_periods() == 366

    def test_warmup_periods_default(self):
        """기본값의 warmup_periods."""
        config = GKBreakoutConfig()
        # max(20*2, 20, 30, 14) + 1 = max(40, 20, 30, 14) + 1 = 41
        assert config.warmup_periods() == 41

    def test_hedge_threshold_range(self):
        """hedge_threshold 범위 검증 (-0.30 ~ -0.05)."""
        GKBreakoutConfig(hedge_threshold=-0.30)
        GKBreakoutConfig(hedge_threshold=-0.05)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(hedge_threshold=-0.31)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(hedge_threshold=-0.04)

    def test_hedge_strength_ratio_range(self):
        """hedge_strength_ratio 범위 검증 (0.1 ~ 1.0)."""
        GKBreakoutConfig(hedge_strength_ratio=0.1)
        GKBreakoutConfig(hedge_strength_ratio=1.0)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(hedge_strength_ratio=0.09)

        with pytest.raises(ValidationError):
            GKBreakoutConfig(hedge_strength_ratio=1.01)


class TestShortMode:
    """ShortMode enum 테스트."""

    def test_values(self):
        """ShortMode 값 검증."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode 사용 가능."""
        config = GKBreakoutConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = GKBreakoutConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED

        config = GKBreakoutConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY
