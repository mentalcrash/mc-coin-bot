"""Tests for HARVolConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.har_vol.config import HARVolConfig
from src.strategy.tsmom.config import ShortMode


class TestHARVolConfigDefaults:
    """기본값 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = HARVolConfig()

        assert config.daily_window == 1
        assert config.weekly_window == 5
        assert config.monthly_window == 22
        assert config.training_window == 252
        assert config.vol_surprise_threshold == 0.0
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = HARVolConfig()

        with pytest.raises(ValidationError):
            config.daily_window = 3  # type: ignore[misc]


class TestHARVolConfigValidation:
    """설정 검증 테스트."""

    def test_daily_window_range(self):
        """daily_window 범위 검증."""
        config = HARVolConfig(daily_window=1)
        assert config.daily_window == 1

        config = HARVolConfig(daily_window=5)
        assert config.daily_window == 5

        with pytest.raises(ValidationError):
            HARVolConfig(daily_window=0)

        with pytest.raises(ValidationError):
            HARVolConfig(daily_window=6)

    def test_weekly_window_range(self):
        """weekly_window 범위 검증."""
        config = HARVolConfig(weekly_window=3)
        assert config.weekly_window == 3

        config = HARVolConfig(weekly_window=10)
        assert config.weekly_window == 10

        with pytest.raises(ValidationError):
            HARVolConfig(weekly_window=2)

        with pytest.raises(ValidationError):
            HARVolConfig(weekly_window=11)

    def test_monthly_window_range(self):
        """monthly_window 범위 검증."""
        config = HARVolConfig(monthly_window=15)
        assert config.monthly_window == 15

        config = HARVolConfig(monthly_window=30)
        assert config.monthly_window == 30

        with pytest.raises(ValidationError):
            HARVolConfig(monthly_window=14)

        with pytest.raises(ValidationError):
            HARVolConfig(monthly_window=31)

    def test_training_window_range(self):
        """training_window 범위 검증."""
        config = HARVolConfig(training_window=60, monthly_window=15)
        assert config.training_window == 60

        config = HARVolConfig(training_window=504)
        assert config.training_window == 504

        with pytest.raises(ValidationError):
            HARVolConfig(training_window=59)

        with pytest.raises(ValidationError):
            HARVolConfig(training_window=505)

    def test_training_gt_monthly_validation(self):
        """training_window > monthly_window 검증."""
        # 유효한 경우
        config = HARVolConfig(training_window=60, monthly_window=15)
        assert config.training_window > config.monthly_window

        # training_window == monthly_window는 에러 (must be >)
        with pytest.raises(ValidationError, match="training_window"):
            HARVolConfig(training_window=22, monthly_window=22)

        # training_window < monthly_window도 에러
        # NOTE: field range (training_window ge=60, monthly_window le=30) 때문에
        # 정상 범위 내에서는 항상 training > monthly이지만,
        # 경계값에서 검증이 작동하는지 확인
        with pytest.raises(ValidationError):
            HARVolConfig(training_window=15, monthly_window=22)

    def test_vol_target_validation(self):
        """vol_target >= min_volatility 검증."""
        # 유효한 경우
        config = HARVolConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # vol_target < min_volatility는 에러
        with pytest.raises(ValidationError, match="vol_target"):
            HARVolConfig(vol_target=0.05, min_volatility=0.10)

    def test_short_mode(self):
        """short_mode 설정 테스트."""
        config = HARVolConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = HARVolConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED

        config = HARVolConfig(short_mode=ShortMode.HEDGE_ONLY)
        assert config.short_mode == ShortMode.HEDGE_ONLY


class TestWarmupPeriods:
    """warmup_periods 테스트."""

    def test_warmup_default(self):
        """기본값으로 warmup 계산."""
        config = HARVolConfig()
        # 252 + 22 + 1 = 275
        assert config.warmup_periods() == 275

    def test_warmup_custom(self):
        """커스텀 파라미터로 warmup 계산."""
        config = HARVolConfig(
            training_window=120,
            monthly_window=20,
        )
        # 120 + 20 + 1 = 141
        assert config.warmup_periods() == 141
