"""Tests for ZScoreMRConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.zscore_mr.config import ShortMode, ZScoreMRConfig


class TestZScoreMRConfig:
    """ZScoreMRConfig 테스트."""

    def test_default_values(self):
        """기본값이 올바르게 설정되는지 확인."""
        config = ZScoreMRConfig()

        assert config.short_lookback == 20
        assert config.long_lookback == 60
        assert config.entry_z == 2.0
        assert config.exit_z == 0.5
        assert config.vol_regime_lookback == 20
        assert config.vol_rank_lookback == 252
        assert config.high_vol_percentile == 0.7
        assert config.vol_window == 30
        assert config.vol_target == 0.20
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.FULL
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        """frozen 모델이므로 속성 수정 불가."""
        config = ZScoreMRConfig()

        with pytest.raises(ValidationError):
            config.short_lookback = 30  # type: ignore[misc]

    def test_short_lookback_range(self):
        """short_lookback 범위 검증 (5-100)."""
        ZScoreMRConfig(short_lookback=5, long_lookback=60)
        ZScoreMRConfig(short_lookback=50, long_lookback=60)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(short_lookback=4)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(short_lookback=101)

    def test_long_lookback_range(self):
        """long_lookback 범위 검증 (20-365)."""
        ZScoreMRConfig(long_lookback=21, short_lookback=20)
        ZScoreMRConfig(long_lookback=365, short_lookback=20)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(long_lookback=19)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(long_lookback=366)

    def test_entry_z_greater_than_exit_z(self):
        """entry_z > exit_z 검증."""
        ZScoreMRConfig(entry_z=2.0, exit_z=0.5)
        ZScoreMRConfig(entry_z=1.0, exit_z=0.5)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(entry_z=0.5, exit_z=0.5)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(entry_z=0.5, exit_z=1.0)

    def test_long_lookback_greater_than_short_lookback(self):
        """long_lookback > short_lookback 검증."""
        ZScoreMRConfig(short_lookback=20, long_lookback=60)
        ZScoreMRConfig(short_lookback=20, long_lookback=21)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(short_lookback=60, long_lookback=60)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(short_lookback=60, long_lookback=30)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        ZScoreMRConfig(vol_target=0.20, min_volatility=0.05)
        ZScoreMRConfig(vol_target=0.05, min_volatility=0.05)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(vol_target=0.04, min_volatility=0.05)

    def test_entry_z_range(self):
        """entry_z 범위 검증 (0.5-4.0)."""
        ZScoreMRConfig(entry_z=0.6, exit_z=0.5)
        ZScoreMRConfig(entry_z=4.0, exit_z=0.5)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(entry_z=0.4)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(entry_z=4.1)

    def test_exit_z_range(self):
        """exit_z 범위 검증 (0.0-2.0)."""
        ZScoreMRConfig(entry_z=2.5, exit_z=0.0)
        ZScoreMRConfig(entry_z=2.5, exit_z=2.0)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(exit_z=-0.1)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(exit_z=2.1)

    def test_high_vol_percentile_range(self):
        """high_vol_percentile 범위 검증 (0.3-0.9)."""
        ZScoreMRConfig(high_vol_percentile=0.3)
        ZScoreMRConfig(high_vol_percentile=0.9)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(high_vol_percentile=0.2)

        with pytest.raises(ValidationError):
            ZScoreMRConfig(high_vol_percentile=0.91)

    def test_warmup_periods(self):
        """warmup_periods가 최대 기간 + 1을 반환."""
        config = ZScoreMRConfig()
        # max(long_lookback=60, vol_rank_lookback=252, vol_window=30, atr_period=14) + 1
        assert config.warmup_periods() == 253

    def test_warmup_periods_custom(self):
        """커스텀 설정에서 warmup_periods 계산."""
        config = ZScoreMRConfig(
            short_lookback=10,
            long_lookback=30,
            vol_rank_lookback=60,
            vol_window=100,
            atr_period=14,
        )
        # max(30, 60, 100, 14) + 1 = 101
        assert config.warmup_periods() == 101

    def test_for_timeframe_daily(self):
        """for_timeframe('1d') 검증."""
        config = ZScoreMRConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_hourly(self):
        """for_timeframe('1h') 검증."""
        config = ZScoreMRConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_with_overrides(self):
        """for_timeframe에 kwargs 오버라이드."""
        config = ZScoreMRConfig.for_timeframe("4h", vol_target=0.30)
        assert config.annualization_factor == 2190.0
        assert config.vol_target == 0.30


class TestShortMode:
    """ShortMode enum 테스트."""

    def test_values(self):
        """ShortMode 값 검증."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode 사용 가능."""
        config = ZScoreMRConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = ZScoreMRConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY

        config = ZScoreMRConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED

    def test_default_is_full(self):
        """평균회귀 전략의 기본 ShortMode는 FULL."""
        config = ZScoreMRConfig()
        assert config.short_mode == ShortMode.FULL
