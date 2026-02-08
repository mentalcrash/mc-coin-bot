"""Unit tests for RiskMomConfig."""

import pytest
from pydantic import ValidationError

from src.strategy.risk_mom.config import RiskMomConfig
from src.strategy.tsmom.config import ShortMode


class TestRiskMomConfigDefaults:
    """RiskMomConfig 기본값 테스트."""

    def test_default_values(self) -> None:
        """기본 파라미터가 올바르게 설정되는지 확인."""
        config = RiskMomConfig()
        assert config.lookback == 30
        assert config.var_window == 126
        assert config.vol_target == 0.30
        assert config.short_mode == ShortMode.HEDGE_ONLY

    def test_frozen_model(self) -> None:
        """frozen=True로 인해 속성 변경이 불가한지 확인."""
        config = RiskMomConfig()
        with pytest.raises(ValidationError):
            config.lookback = 50  # type: ignore[misc]

    def test_vol_target_validation(self) -> None:
        """vol_target < min_volatility일 때 ValidationError 발생."""
        with pytest.raises(ValidationError, match="vol_target"):
            RiskMomConfig(vol_target=0.03, min_volatility=0.05)

    def test_vol_target_equal_min_volatility(self) -> None:
        """vol_target == min_volatility일 때 통과."""
        config = RiskMomConfig(vol_target=0.05, min_volatility=0.05)
        assert config.vol_target == config.min_volatility

    def test_vol_target_range_validation(self) -> None:
        """vol_target 범위 검증 (0.05 ~ 1.0)."""
        with pytest.raises(ValidationError):
            RiskMomConfig(vol_target=0.01)
        with pytest.raises(ValidationError):
            RiskMomConfig(vol_target=1.5)


class TestRiskMomConfigWarmup:
    """warmup_periods 계산 테스트."""

    def test_warmup_periods(self) -> None:
        """warmup = max(lookback, var_window, vol_window) + 1."""
        config = RiskMomConfig()
        expected = max(config.lookback, config.var_window, config.vol_window) + 1
        assert config.warmup_periods() == expected

    def test_warmup_periods_lookback_dominant(self) -> None:
        """lookback이 가장 클 때 warmup 계산."""
        config = RiskMomConfig(lookback=200, var_window=100, vol_window=60)
        assert config.warmup_periods() == 201

    def test_warmup_periods_var_window_dominant(self) -> None:
        """var_window이 가장 클 때 warmup 계산 (기본값)."""
        config = RiskMomConfig(lookback=30, var_window=365, vol_window=30)
        assert config.warmup_periods() == 366


class TestRiskMomConfigTimeframe:
    """for_timeframe() 팩토리 메서드 테스트."""

    def test_for_timeframe_1d(self) -> None:
        """1d 타임프레임에서 annualization_factor=365.0."""
        config = RiskMomConfig.for_timeframe("1d")
        assert config.annualization_factor == 365.0
        assert config.lookback == 7

    def test_for_timeframe_1h(self) -> None:
        """1h 타임프레임에서 annualization_factor=8760.0."""
        config = RiskMomConfig.for_timeframe("1h")
        assert config.annualization_factor == 8760.0
        assert config.lookback == 24

    def test_for_timeframe_with_override(self) -> None:
        """for_timeframe에 추가 파라미터 오버라이드."""
        config = RiskMomConfig.for_timeframe("1d", vol_target=0.50)
        assert config.annualization_factor == 365.0
        assert config.vol_target == 0.50


class TestRiskMomConfigPresets:
    """preset 팩토리 메서드 테스트."""

    def test_conservative_preset(self) -> None:
        """보수적 설정: 긴 lookback, 낮은 vol_target."""
        config = RiskMomConfig.conservative()
        assert config.lookback == 48
        assert config.vol_window == 48
        assert config.var_window == 180
        assert config.vol_target == 0.10
        assert config.min_volatility == 0.08

    def test_aggressive_preset(self) -> None:
        """공격적 설정: 짧은 lookback, 높은 vol_target."""
        config = RiskMomConfig.aggressive()
        assert config.lookback == 12
        assert config.vol_window == 12
        assert config.var_window == 63
        assert config.vol_target == 0.20
        assert config.min_volatility == 0.05


class TestRiskMomConfigEdgeCases:
    """경계값 및 엣지 케이스 테스트."""

    def test_lookback_min_boundary(self) -> None:
        """lookback 최소값 (6)."""
        config = RiskMomConfig(lookback=6, vol_window=6)
        assert config.lookback == 6

    def test_lookback_below_min_raises(self) -> None:
        """lookback < 6일 때 ValidationError."""
        with pytest.raises(ValidationError):
            RiskMomConfig(lookback=5)

    def test_var_window_min_boundary(self) -> None:
        """var_window 최소값 (60)."""
        config = RiskMomConfig(var_window=60)
        assert config.var_window == 60

    def test_var_window_below_min_raises(self) -> None:
        """var_window < 60일 때 ValidationError."""
        with pytest.raises(ValidationError):
            RiskMomConfig(var_window=59)

    def test_hedge_threshold_range(self) -> None:
        """hedge_threshold 범위 (-0.30 ~ -0.05)."""
        config = RiskMomConfig(hedge_threshold=-0.10)
        assert config.hedge_threshold == -0.10

        with pytest.raises(ValidationError):
            RiskMomConfig(hedge_threshold=-0.01)
        with pytest.raises(ValidationError):
            RiskMomConfig(hedge_threshold=-0.50)
