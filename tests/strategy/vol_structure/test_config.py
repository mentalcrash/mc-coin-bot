"""Tests for VolStructureConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.vol_structure.config import ShortMode, VolStructureConfig


class TestShortMode:
    """ShortMode IntEnum 테스트."""

    def test_values(self):
        """ShortMode 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2

    def test_config_accepts_short_mode(self):
        """Config에서 ShortMode를 올바르게 수용."""
        config = VolStructureConfig(short_mode=ShortMode.FULL)
        assert config.short_mode == ShortMode.FULL

        config = VolStructureConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED


class TestVolStructureConfig:
    """VolStructureConfig 테스트."""

    def test_default_values(self):
        """기본값으로 생성 테스트."""
        config = VolStructureConfig()

        assert config.vol_short_window == 10
        assert config.vol_long_window == 60
        assert config.mom_window == 20
        assert config.expansion_vol_ratio == 1.2
        assert config.contraction_vol_ratio == 0.8
        assert config.expansion_mom_threshold == 1.5
        assert config.contraction_mom_threshold == 0.5
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.use_log_returns is True
        assert config.atr_period == 14
        assert config.short_mode == ShortMode.HEDGE_ONLY
        assert config.hedge_threshold == -0.07
        assert config.hedge_strength_ratio == 0.8

    def test_frozen_model(self):
        """Frozen 모델이므로 변경 불가."""
        config = VolStructureConfig()

        with pytest.raises(ValidationError):
            config.vol_short_window = 20  # type: ignore[misc]

    def test_vol_short_window_range(self):
        """vol_short_window 범위 검증."""
        config = VolStructureConfig(vol_short_window=5)
        assert config.vol_short_window == 5

        config = VolStructureConfig(vol_short_window=30, vol_long_window=60)
        assert config.vol_short_window == 30

        with pytest.raises(ValidationError):
            VolStructureConfig(vol_short_window=4)

        with pytest.raises(ValidationError):
            VolStructureConfig(vol_short_window=31)

    def test_vol_long_window_range(self):
        """vol_long_window 범위 검증."""
        config = VolStructureConfig(vol_long_window=30, vol_short_window=5)
        assert config.vol_long_window == 30

        config = VolStructureConfig(vol_long_window=120)
        assert config.vol_long_window == 120

        with pytest.raises(ValidationError):
            VolStructureConfig(vol_long_window=29)

        with pytest.raises(ValidationError):
            VolStructureConfig(vol_long_window=121)

    def test_mom_window_range(self):
        """mom_window 범위 검증."""
        config = VolStructureConfig(mom_window=10)
        assert config.mom_window == 10

        config = VolStructureConfig(mom_window=60)
        assert config.mom_window == 60

        with pytest.raises(ValidationError):
            VolStructureConfig(mom_window=9)

        with pytest.raises(ValidationError):
            VolStructureConfig(mom_window=61)

    def test_long_greater_than_short_validation(self):
        """vol_long_window > vol_short_window 검증."""
        # 유효한 경우
        config = VolStructureConfig(vol_short_window=10, vol_long_window=60)
        assert config.vol_long_window > config.vol_short_window

        # long <= short는 에러
        with pytest.raises(ValidationError):
            VolStructureConfig(vol_short_window=30, vol_long_window=30)

    def test_expansion_gt_contraction_vol_ratio(self):
        """expansion_vol_ratio > contraction_vol_ratio 검증."""
        # 유효한 경우
        config = VolStructureConfig(expansion_vol_ratio=1.5, contraction_vol_ratio=0.5)
        assert config.expansion_vol_ratio > config.contraction_vol_ratio

        # expansion <= contraction은 에러
        with pytest.raises(ValidationError):
            VolStructureConfig(expansion_vol_ratio=1.0, contraction_vol_ratio=1.0)

        with pytest.raises(ValidationError):
            VolStructureConfig(expansion_vol_ratio=1.0, contraction_vol_ratio=1.0)

    def test_expansion_gt_contraction_mom_threshold(self):
        """expansion_mom_threshold > contraction_mom_threshold 검증."""
        # 유효한 경우
        config = VolStructureConfig(expansion_mom_threshold=2.0, contraction_mom_threshold=0.3)
        assert config.expansion_mom_threshold > config.contraction_mom_threshold

        # expansion <= contraction은 에러
        with pytest.raises(ValidationError):
            VolStructureConfig(expansion_mom_threshold=0.5, contraction_mom_threshold=0.5)

    def test_vol_target_gte_min_volatility(self):
        """vol_target >= min_volatility 검증."""
        # 유효한 경우
        config = VolStructureConfig(vol_target=0.10, min_volatility=0.05)
        assert config.vol_target >= config.min_volatility

        # vol_target < min_volatility는 에러
        with pytest.raises(ValidationError):
            VolStructureConfig(vol_target=0.05, min_volatility=0.10)

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        config = VolStructureConfig(
            vol_short_window=10,
            vol_long_window=60,
            mom_window=20,
            atr_period=14,
        )
        # max(60, 20, 14) + 1 = 61
        assert config.warmup_periods() == 61

    def test_warmup_periods_custom(self):
        """커스텀 파라미터로 warmup_periods() 테스트."""
        config = VolStructureConfig(
            vol_short_window=10,
            vol_long_window=80,
            mom_window=60,
            atr_period=50,
        )
        # max(80, 60, 50) + 1 = 81
        assert config.warmup_periods() == 81
