"""Tests for FundingCarryConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.funding_carry.config import FundingCarryConfig
from src.strategy.tsmom.config import ShortMode


class TestFundingCarryConfigDefaults:
    """기본값 테스트."""

    def test_default_values(self) -> None:
        """기본값으로 생성 테스트."""
        config = FundingCarryConfig()

        assert config.lookback == 3
        assert config.zscore_window == 90
        assert config.vol_window == 30
        assert config.vol_target == 0.35
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.entry_threshold == 0.0001
        assert config.use_log_returns is True
        assert config.short_mode == ShortMode.FULL

    def test_frozen_model(self) -> None:
        """Frozen 모델이므로 변경 불가."""
        config = FundingCarryConfig()

        with pytest.raises(ValidationError):
            config.lookback = 5  # type: ignore[misc]


class TestFundingCarryConfigRanges:
    """필드 범위 검증 테스트."""

    def test_lookback_range_valid(self) -> None:
        """lookback 유효 범위 테스트."""
        config = FundingCarryConfig(lookback=1)
        assert config.lookback == 1

        config = FundingCarryConfig(lookback=30)
        assert config.lookback == 30

    def test_lookback_range_invalid(self) -> None:
        """lookback 범위 초과 시 에러."""
        with pytest.raises(ValidationError):
            FundingCarryConfig(lookback=0)

        with pytest.raises(ValidationError):
            FundingCarryConfig(lookback=31)

    def test_zscore_window_range_valid(self) -> None:
        """zscore_window 유효 범위 테스트."""
        config = FundingCarryConfig(zscore_window=10)
        assert config.zscore_window == 10

        config = FundingCarryConfig(zscore_window=365)
        assert config.zscore_window == 365

    def test_zscore_window_range_invalid(self) -> None:
        """zscore_window 범위 초과 시 에러."""
        with pytest.raises(ValidationError):
            FundingCarryConfig(zscore_window=9)

        with pytest.raises(ValidationError):
            FundingCarryConfig(zscore_window=366)

    def test_entry_threshold_range_valid(self) -> None:
        """entry_threshold 유효 범위 테스트."""
        config = FundingCarryConfig(entry_threshold=0.0)
        assert config.entry_threshold == 0.0

        config = FundingCarryConfig(entry_threshold=0.01)
        assert config.entry_threshold == 0.01

    def test_entry_threshold_range_invalid(self) -> None:
        """entry_threshold 범위 초과 시 에러."""
        with pytest.raises(ValidationError):
            FundingCarryConfig(entry_threshold=-0.001)

        with pytest.raises(ValidationError):
            FundingCarryConfig(entry_threshold=0.02)

    def test_vol_window_range(self) -> None:
        """vol_window 범위 검증."""
        config = FundingCarryConfig(vol_window=5)
        assert config.vol_window == 5

        config = FundingCarryConfig(vol_window=120)
        assert config.vol_window == 120

        with pytest.raises(ValidationError):
            FundingCarryConfig(vol_window=4)

        with pytest.raises(ValidationError):
            FundingCarryConfig(vol_window=121)


class TestFundingCarryConfigValidation:
    """모델 검증 테스트."""

    def test_vol_target_gte_min_volatility(self) -> None:
        """vol_target >= min_volatility 검증."""
        config = FundingCarryConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

    def test_vol_target_lt_min_volatility_raises(self) -> None:
        """vol_target < min_volatility일 때 에러."""
        with pytest.raises(ValidationError):
            FundingCarryConfig(vol_target=0.05, min_volatility=0.10)

    def test_short_mode_full(self) -> None:
        """기본 short_mode가 FULL인지 확인 (carry 전략)."""
        config = FundingCarryConfig()
        assert config.short_mode == ShortMode.FULL

    def test_short_mode_disabled(self) -> None:
        """DISABLED 모드 설정."""
        config = FundingCarryConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestFundingCarryConfigWarmup:
    """워밍업 기간 테스트."""

    def test_warmup_default(self) -> None:
        """기본 설정의 warmup_periods."""
        config = FundingCarryConfig()
        # max(90, 30) + 1 = 91
        assert config.warmup_periods() == 91

    def test_warmup_custom(self) -> None:
        """커스텀 설정의 warmup_periods."""
        config = FundingCarryConfig(zscore_window=120, vol_window=60)
        # max(120, 60) + 1 = 121
        assert config.warmup_periods() == 121

    def test_warmup_vol_window_dominant(self) -> None:
        """vol_window이 더 클 때 warmup_periods."""
        config = FundingCarryConfig(zscore_window=10, vol_window=120)
        # max(10, 120) + 1 = 121
        assert config.warmup_periods() == 121
