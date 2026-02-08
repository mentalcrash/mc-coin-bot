"""Tests for VolRegimeConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.vol_regime.config import ShortMode, VolRegimeConfig


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_int_conversion(self):
        """ShortMode -> int 변환."""
        assert int(ShortMode.DISABLED) == 0
        assert int(ShortMode.HEDGE_ONLY) == 1
        assert int(ShortMode.FULL) == 2


class TestVolRegimeConfig:
    """VolRegimeConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = VolRegimeConfig()

        assert config.vol_lookback == 20
        assert config.vol_rank_lookback == 252
        assert config.high_vol_threshold == 0.8
        assert config.low_vol_threshold == 0.2
        assert config.high_vol_lookback == 60
        assert config.high_vol_target == 0.15
        assert config.normal_lookback == 30
        assert config.normal_vol_target == 0.30
        assert config.low_vol_lookback == 14
        assert config.low_vol_target == 0.50
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = VolRegimeConfig()

        with pytest.raises(ValidationError):
            config.vol_lookback = 30  # type: ignore[misc]

    def test_threshold_ordering_validation(self):
        """high_vol_threshold > low_vol_threshold 검증."""
        # 유효한 경우
        config = VolRegimeConfig(high_vol_threshold=0.7, low_vol_threshold=0.3)
        assert config.high_vol_threshold > config.low_vol_threshold

        # high <= low는 에러
        with pytest.raises(ValidationError):
            VolRegimeConfig(high_vol_threshold=0.5, low_vol_threshold=0.5)

        with pytest.raises(ValidationError):
            VolRegimeConfig(high_vol_threshold=0.3, low_vol_threshold=0.4)

    def test_vol_target_vs_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        # 유효한 경우
        config = VolRegimeConfig(
            high_vol_target=0.10,
            normal_vol_target=0.20,
            low_vol_target=0.30,
            min_volatility=0.05,
        )
        assert config.high_vol_target >= config.min_volatility

        # high_vol_target < min_volatility는 에러
        with pytest.raises(ValidationError):
            VolRegimeConfig(high_vol_target=0.03, min_volatility=0.05)

    def test_vol_lookback_range(self):
        """vol_lookback 범위 검증."""
        config = VolRegimeConfig(vol_lookback=5)
        assert config.vol_lookback == 5

        config = VolRegimeConfig(vol_lookback=60)
        assert config.vol_lookback == 60

        with pytest.raises(ValidationError):
            VolRegimeConfig(vol_lookback=4)

        with pytest.raises(ValidationError):
            VolRegimeConfig(vol_lookback=61)

    def test_vol_rank_lookback_range(self):
        """vol_rank_lookback 범위 검증."""
        config = VolRegimeConfig(vol_rank_lookback=60)
        assert config.vol_rank_lookback == 60

        config = VolRegimeConfig(vol_rank_lookback=500)
        assert config.vol_rank_lookback == 500

        with pytest.raises(ValidationError):
            VolRegimeConfig(vol_rank_lookback=59)

        with pytest.raises(ValidationError):
            VolRegimeConfig(vol_rank_lookback=501)

    def test_high_vol_threshold_range(self):
        """high_vol_threshold 범위 검증."""
        config = VolRegimeConfig(high_vol_threshold=0.5)
        assert config.high_vol_threshold == 0.5

        config = VolRegimeConfig(high_vol_threshold=0.95)
        assert config.high_vol_threshold == 0.95

        with pytest.raises(ValidationError):
            VolRegimeConfig(high_vol_threshold=0.49)

        with pytest.raises(ValidationError):
            VolRegimeConfig(high_vol_threshold=0.96)

    def test_low_vol_threshold_range(self):
        """low_vol_threshold 범위 검증."""
        config = VolRegimeConfig(low_vol_threshold=0.05)
        assert config.low_vol_threshold == 0.05

        config = VolRegimeConfig(low_vol_threshold=0.5, high_vol_threshold=0.6)
        assert config.low_vol_threshold == 0.5

        with pytest.raises(ValidationError):
            VolRegimeConfig(low_vol_threshold=0.04)

        with pytest.raises(ValidationError):
            VolRegimeConfig(low_vol_threshold=0.51)

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = VolRegimeConfig(
            vol_rank_lookback=252,
            high_vol_lookback=60,
            normal_lookback=30,
            atr_period=14,
        )
        # max(252, 60, 30, 14) + 1 = 253
        assert config.warmup_periods() == 253

    def test_warmup_periods_custom(self):
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = VolRegimeConfig(
            vol_rank_lookback=100,
            high_vol_lookback=120,
            normal_lookback=90,
            atr_period=50,
        )
        # max(100, 120, 90, 50) + 1 = 121
        assert config.warmup_periods() == 121

    def test_custom_params(self):
        """커스텀 파라미터로 생성 테스트."""
        config = VolRegimeConfig(
            vol_lookback=30,
            vol_rank_lookback=120,
            high_vol_threshold=0.75,
            low_vol_threshold=0.25,
            high_vol_lookback=90,
            high_vol_target=0.10,
            normal_lookback=45,
            normal_vol_target=0.25,
            low_vol_lookback=20,
            low_vol_target=0.40,
            short_mode=ShortMode.FULL,
        )

        assert config.vol_lookback == 30
        assert config.vol_rank_lookback == 120
        assert config.high_vol_threshold == 0.75
        assert config.low_vol_threshold == 0.25
        assert config.short_mode == ShortMode.FULL
