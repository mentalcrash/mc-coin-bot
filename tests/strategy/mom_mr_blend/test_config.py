"""Unit tests for MomMrBlendConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.mom_mr_blend.config import MomMrBlendConfig, ShortMode


class TestMomMrBlendConfigDefaults:
    """MomMrBlendConfig 기본값 테스트."""

    def test_default_values(self) -> None:
        """기본 파라미터가 올바르게 설정되는지 확인."""
        config = MomMrBlendConfig()
        assert config.mom_lookback == 28
        assert config.mom_z_window == 90
        assert config.mr_lookback == 14
        assert config.mr_z_window == 90
        assert config.mom_weight == 0.5
        assert config.mr_weight == 0.5
        assert config.vol_window == 20
        assert config.vol_target == 0.40
        assert config.min_volatility == 0.05
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.DISABLED

    def test_frozen_model(self) -> None:
        """frozen=True로 인해 속성 변경이 불가한지 확인."""
        config = MomMrBlendConfig()
        with pytest.raises(ValidationError):
            config.mom_lookback = 50  # type: ignore[misc]

    def test_short_mode_enum_values(self) -> None:
        """ShortMode 열거형 값 확인."""
        assert ShortMode.DISABLED == 0
        assert ShortMode.HEDGE_ONLY == 1
        assert ShortMode.FULL == 2


class TestMomMrBlendConfigWeightValidation:
    """가중치 검증 테스트."""

    def test_zero_weights_raises(self) -> None:
        """mom_weight + mr_weight == 0일 때 ValidationError."""
        with pytest.raises(ValidationError, match="must be > 0"):
            MomMrBlendConfig(mom_weight=0.0, mr_weight=0.0)

    def test_excessive_weights_raises(self) -> None:
        """mom_weight + mr_weight > 1.01일 때 ValidationError."""
        with pytest.raises(ValidationError, match=r"should be <= 1\.0"):
            MomMrBlendConfig(mom_weight=0.7, mr_weight=0.5)

    def test_equal_weights_valid(self) -> None:
        """50/50 가중치 검증."""
        config = MomMrBlendConfig(mom_weight=0.5, mr_weight=0.5)
        assert config.mom_weight == 0.5
        assert config.mr_weight == 0.5

    def test_asymmetric_weights_valid(self) -> None:
        """비대칭 가중치 (합이 1.0 이하) 검증."""
        config = MomMrBlendConfig(mom_weight=0.7, mr_weight=0.3)
        assert config.mom_weight == 0.7
        assert config.mr_weight == 0.3

    def test_single_weight_valid(self) -> None:
        """한쪽 가중치만 있는 경우 (합 > 0)."""
        config = MomMrBlendConfig(mom_weight=1.0, mr_weight=0.0)
        assert config.mom_weight == 1.0
        assert config.mr_weight == 0.0

    def test_vol_target_below_min_raises(self) -> None:
        """vol_target < min_volatility일 때 ValidationError."""
        with pytest.raises(ValidationError, match="vol_target"):
            MomMrBlendConfig(vol_target=0.03, min_volatility=0.05)

    def test_vol_target_equal_min_volatility(self) -> None:
        """vol_target == min_volatility일 때 통과."""
        config = MomMrBlendConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target == config.min_volatility


class TestMomMrBlendConfigWarmup:
    """warmup_periods 계산 테스트."""

    def test_warmup_periods_default(self) -> None:
        """기본 설정에서 warmup = max(28+90, 14+90, 20) + 1 = 119."""
        config = MomMrBlendConfig()
        expected = max(28 + 90, 14 + 90, 20) + 1  # 118 + 1 = 119
        assert config.warmup_periods() == expected

    def test_warmup_periods_mom_dominant(self) -> None:
        """mom_lookback + mom_z_window이 최대일 때."""
        config = MomMrBlendConfig(
            mom_lookback=120, mom_z_window=365, mr_lookback=14, mr_z_window=90
        )
        expected = max(120 + 365, 14 + 90, config.vol_window) + 1
        assert config.warmup_periods() == expected

    def test_warmup_periods_mr_dominant(self) -> None:
        """mr_lookback + mr_z_window이 최대일 때."""
        config = MomMrBlendConfig(mom_lookback=5, mom_z_window=20, mr_lookback=60, mr_z_window=365)
        expected = max(5 + 20, 60 + 365, config.vol_window) + 1
        assert config.warmup_periods() == expected


class TestMomMrBlendConfigTimeframe:
    """for_timeframe() 팩토리 메서드 테스트."""

    def test_for_timeframe_1d(self) -> None:
        """1d 타임프레임에서 annualization_factor=365.0."""
        config = MomMrBlendConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_1h(self) -> None:
        """1h 타임프레임에서 annualization_factor=8760.0."""
        config = MomMrBlendConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0

    def test_for_timeframe_4h(self) -> None:
        """4h 타임프레임에서 annualization_factor=2190.0."""
        config = MomMrBlendConfig.for_timeframe("4h")
        assert config.annualization_factor == 2190.0

    def test_for_timeframe_unknown_defaults_365(self) -> None:
        """알 수 없는 타임프레임에서 기본 365.0."""
        config = MomMrBlendConfig.for_timeframe("2d")
        assert config.annualization_factor == 365.0

    def test_for_timeframe_with_override(self) -> None:
        """for_timeframe에 추가 파라미터 오버라이드."""
        config = MomMrBlendConfig.for_timeframe("1d", vol_target=0.50)
        assert config.annualization_factor == 365.0
        assert config.vol_target == 0.50


class TestMomMrBlendConfigPresets:
    """preset 팩토리 메서드 테스트."""

    def test_conservative_preset(self) -> None:
        """보수적 설정: 긴 lookback, 낮은 vol_target."""
        config = MomMrBlendConfig.conservative()
        assert config.mom_lookback == 42
        assert config.mr_lookback == 21
        assert config.vol_target == 0.30

    def test_aggressive_preset(self) -> None:
        """공격적 설정: 짧은 lookback, 높은 vol_target."""
        config = MomMrBlendConfig.aggressive()
        assert config.mom_lookback == 14
        assert config.mr_lookback == 7
        assert config.vol_target == 0.50


class TestMomMrBlendConfigEdgeCases:
    """경계값 및 엣지 케이스 테스트."""

    def test_mom_lookback_min_boundary(self) -> None:
        """mom_lookback 최소값 (5)."""
        config = MomMrBlendConfig(mom_lookback=5)
        assert config.mom_lookback == 5

    def test_mom_lookback_below_min_raises(self) -> None:
        """mom_lookback < 5일 때 ValidationError."""
        with pytest.raises(ValidationError):
            MomMrBlendConfig(mom_lookback=4)

    def test_mom_lookback_max_boundary(self) -> None:
        """mom_lookback 최대값 (120)."""
        config = MomMrBlendConfig(mom_lookback=120)
        assert config.mom_lookback == 120

    def test_mom_lookback_above_max_raises(self) -> None:
        """mom_lookback > 120일 때 ValidationError."""
        with pytest.raises(ValidationError):
            MomMrBlendConfig(mom_lookback=121)

    def test_mr_lookback_min_boundary(self) -> None:
        """mr_lookback 최소값 (3)."""
        config = MomMrBlendConfig(mr_lookback=3)
        assert config.mr_lookback == 3

    def test_mr_lookback_below_min_raises(self) -> None:
        """mr_lookback < 3일 때 ValidationError."""
        with pytest.raises(ValidationError):
            MomMrBlendConfig(mr_lookback=2)

    def test_vol_target_range_validation(self) -> None:
        """vol_target 범위 검증 (0.05 ~ 1.0)."""
        with pytest.raises(ValidationError):
            MomMrBlendConfig(vol_target=0.01)
        with pytest.raises(ValidationError):
            MomMrBlendConfig(vol_target=1.5)

    def test_vol_window_boundaries(self) -> None:
        """vol_window 범위 검증 (5 ~ 100)."""
        config_min = MomMrBlendConfig(vol_window=5)
        assert config_min.vol_window == 5
        config_max = MomMrBlendConfig(vol_window=100)
        assert config_max.vol_window == 100

        with pytest.raises(ValidationError):
            MomMrBlendConfig(vol_window=4)
        with pytest.raises(ValidationError):
            MomMrBlendConfig(vol_window=101)

    def test_short_mode_from_int(self) -> None:
        """정수로 ShortMode 설정."""
        config = MomMrBlendConfig(short_mode=2)
        assert config.short_mode == ShortMode.FULL
