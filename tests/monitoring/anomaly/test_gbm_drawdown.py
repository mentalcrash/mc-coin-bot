"""Tests for GBM Drawdown Monitor."""

from __future__ import annotations

import math

import pytest

from src.monitoring.anomaly.gbm_drawdown import (
    DrawdownCheckResult,
    DrawdownSeverity,
    GBMDrawdownMonitor,
    _norm_ppf,
)


class TestInitialState:
    """초기 상태 테스트."""

    def test_first_update_returns_normal(self) -> None:
        monitor = GBMDrawdownMonitor(mu=0.0005, sigma=0.02)
        result = monitor.update(0.01)
        assert result.severity == DrawdownSeverity.NORMAL
        assert result.current_depth == 0.0

    def test_early_updates_always_normal(self) -> None:
        """n_days < 30이면 항상 NORMAL."""
        monitor = GBMDrawdownMonitor(mu=0.0005, sigma=0.02)
        for _ in range(29):
            result = monitor.update(-0.05)  # 큰 손실도 NORMAL
        assert result.severity == DrawdownSeverity.NORMAL


class TestNormalDrawdown:
    """정상 drawdown 테스트."""

    def test_small_losses_are_normal(self) -> None:
        """작은 손실은 정상 범위."""
        monitor = GBMDrawdownMonitor(mu=0.0005, sigma=0.02)
        # 30일 warmup: 약간의 변동 포함
        for i in range(35):
            r = 0.005 if i % 2 == 0 else -0.003  # 양/음 혼합
            monitor.update(r)
        # 작은 손실 1일 — sigma=0.02 대비 아주 작은 drawdown
        result = monitor.update(-0.001)
        assert result.severity == DrawdownSeverity.NORMAL

    def test_positive_returns_no_drawdown(self) -> None:
        """양수 수익은 drawdown이 0."""
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        for _ in range(35):
            result = monitor.update(0.01)
        assert result.current_depth == 0.0
        assert result.current_duration_days == 0


class TestAbnormalDepth:
    """비정상 depth 테스트."""

    def test_large_drawdown_triggers_warning(self) -> None:
        """큰 drawdown은 WARNING."""
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.01, confidence=0.95)
        # Warmup with small positive returns
        for _ in range(50):
            monitor.update(0.001)
        # Sudden large losses
        for _ in range(5):
            result = monitor.update(-0.10)

        # depth_exceeded should be True due to extreme losses
        assert result.depth_exceeded is True
        assert result.severity in (DrawdownSeverity.WARNING, DrawdownSeverity.CRITICAL)


class TestAbnormalDuration:
    """비정상 duration 테스트."""

    def test_prolonged_drawdown_triggers_warning(self) -> None:
        """장기 drawdown은 WARNING."""
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.01, confidence=0.95)
        # Push equity up first
        for _ in range(40):
            monitor.update(0.005)
        # Extended small losses
        for _ in range(100):
            result = monitor.update(-0.001)

        # Duration should be exceeding expected
        assert result.current_duration_days > 0
        assert result.duration_exceeded is True


class TestCritical:
    """CRITICAL (depth AND duration) 테스트."""

    def test_critical_requires_both(self) -> None:
        """depth AND duration 초과 시 CRITICAL."""
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.005, confidence=0.95)
        # Warmup
        for _ in range(40):
            monitor.update(0.002)
        # Extreme prolonged losses
        for _ in range(100):
            result = monitor.update(-0.05)

        if result.depth_exceeded and result.duration_exceeded:
            assert result.severity == DrawdownSeverity.CRITICAL


class TestEstimateParams:
    """파라미터 추정 테스트."""

    def test_basic_estimation(self) -> None:
        returns = [0.01, -0.005, 0.008, -0.003, 0.006]
        mu, sigma = GBMDrawdownMonitor.estimate_params(returns)
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_insufficient_data(self) -> None:
        """데이터 부족 시 기본값."""
        mu, sigma = GBMDrawdownMonitor.estimate_params([0.01])
        assert mu == 0.0
        assert sigma == 0.01

    def test_empty_list(self) -> None:
        mu, sigma = GBMDrawdownMonitor.estimate_params([])
        assert mu == 0.0
        assert sigma == 0.01

    def test_filters_nan_inf(self) -> None:
        returns = [0.01, float("nan"), -0.01, float("inf"), 0.005]
        mu, sigma = GBMDrawdownMonitor.estimate_params(returns)
        assert math.isfinite(mu)
        assert math.isfinite(sigma)
        assert sigma > 0


class TestSerialization:
    """to_dict / restore_from_dict 테스트."""

    def test_roundtrip(self) -> None:
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        for _ in range(50):
            monitor.update(0.001)
        monitor.update(-0.05)

        data = monitor.to_dict()

        # Restore into fresh monitor with same params
        restored = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        restored.restore_from_dict(data)

        # Should produce same result
        original_result = monitor.update(0.001)
        restored_result = restored.update(0.001)

        assert original_result.severity == restored_result.severity
        assert abs(original_result.current_depth - restored_result.current_depth) < 1e-10

    def test_empty_restore(self) -> None:
        """빈 dict에서 복원 → 기본값."""
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        monitor.restore_from_dict({})
        result = monitor.update(0.01)
        assert result.severity == DrawdownSeverity.NORMAL


class TestNaNInfGuard:
    """NaN/Inf 입력 guard 테스트."""

    def test_nan_returns_normal(self) -> None:
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        result = monitor.update(float("nan"))
        assert result.severity == DrawdownSeverity.NORMAL

    def test_inf_returns_normal(self) -> None:
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        result = monitor.update(float("inf"))
        assert result.severity == DrawdownSeverity.NORMAL

    def test_neg_inf_returns_normal(self) -> None:
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        result = monitor.update(float("-inf"))
        assert result.severity == DrawdownSeverity.NORMAL


class TestEdgeCases:
    """엣지 케이스 테스트."""

    def test_sigma_zero_guard(self) -> None:
        """sigma=0은 epsilon으로 대체."""
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.0)
        result = monitor.update(0.01)
        assert result.severity == DrawdownSeverity.NORMAL

    def test_n_less_than_30_always_normal(self) -> None:
        """n < 30이면 항상 NORMAL."""
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        for _ in range(29):
            result = monitor.update(-0.1)
            assert result.severity == DrawdownSeverity.NORMAL

    def test_expected_max_drawdown_zero_days(self) -> None:
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        assert monitor.expected_max_drawdown(0) == 0.0

    def test_expected_max_duration_zero_days(self) -> None:
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        assert monitor.expected_max_duration(0) == 0

    def test_reset(self) -> None:
        """reset 후 초기 상태 복원."""
        monitor = GBMDrawdownMonitor(mu=0.001, sigma=0.02)
        for _ in range(50):
            monitor.update(-0.01)
        monitor.reset()
        result = monitor.update(0.01)
        assert result.current_depth == 0.0


class TestNormPpf:
    """_norm_ppf 정확도 테스트."""

    def test_known_values(self) -> None:
        """알려진 z-score 값과 비교."""
        # z(0.95) ≈ 1.6449
        z_95 = _norm_ppf(0.95)
        assert abs(z_95 - 1.6449) < 0.01

        # z(0.975) ≈ 1.9600
        z_975 = _norm_ppf(0.975)
        assert abs(z_975 - 1.9600) < 0.01

        # z(0.5) = 0.0
        z_50 = _norm_ppf(0.5)
        assert abs(z_50) < 0.01

    def test_symmetry(self) -> None:
        """norm_ppf(p) = -norm_ppf(1-p)."""
        z_90 = _norm_ppf(0.9)
        z_10 = _norm_ppf(0.1)
        assert abs(z_90 + z_10) < 0.01

    def test_boundary_values(self) -> None:
        """경계값 (0, 1)은 0.0 반환."""
        assert _norm_ppf(0.0) == 0.0
        assert _norm_ppf(1.0) == 0.0


class TestDrawdownCheckResult:
    """DrawdownCheckResult dataclass 테스트."""

    def test_frozen(self) -> None:
        result = DrawdownCheckResult(
            severity=DrawdownSeverity.NORMAL,
            current_depth=0.05,
            current_duration_days=3,
            expected_max_depth=0.10,
            expected_max_duration=10,
            depth_exceeded=False,
            duration_exceeded=False,
        )
        with pytest.raises(AttributeError):
            result.severity = DrawdownSeverity.WARNING  # type: ignore[misc]
