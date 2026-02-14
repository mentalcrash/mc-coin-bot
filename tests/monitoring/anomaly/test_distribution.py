"""Tests for Distribution Drift Detector."""

from __future__ import annotations

import random

from src.monitoring.anomaly.distribution import (
    DistributionDriftDetector,
    DriftCheckResult,
    DriftSeverity,
)


class TestInitialState:
    """최소 샘플 미달 시 NORMAL 반환."""

    def test_empty_returns_normal(self) -> None:
        detector = DistributionDriftDetector(reference_returns=[0.01] * 100)
        result = detector.update(0.01)
        assert result.severity == DriftSeverity.NORMAL
        assert result.recent_n == 1

    def test_under_30_samples_normal(self) -> None:
        """recent 29개 → NORMAL."""
        detector = DistributionDriftDetector(reference_returns=[0.01] * 100)
        for _ in range(29):
            result = detector.update(0.01)
        assert result.severity == DriftSeverity.NORMAL
        assert result.recent_n == 29

    def test_insufficient_reference_normal(self) -> None:
        """reference 10개 → NORMAL (30 미만)."""
        detector = DistributionDriftDetector(reference_returns=[0.01] * 10)
        for _ in range(50):
            result = detector.update(0.01)
        assert result.severity == DriftSeverity.NORMAL


class TestNormalDistribution:
    """동일 분포 → drift 미감지."""

    def test_same_distribution_no_drift(self) -> None:
        rng = random.Random(42)
        ref = [rng.gauss(0.001, 0.02) for _ in range(200)]
        detector = DistributionDriftDetector(reference_returns=ref)

        # recent도 동일 분포에서 샘플링
        for _ in range(60):
            result = detector.update(rng.gauss(0.001, 0.02))

        assert result.severity == DriftSeverity.NORMAL
        assert result.drifted is False


class TestDriftedDistribution:
    """분포 이동 → WARNING 이상."""

    def test_shifted_mean_detected(self) -> None:
        rng = random.Random(42)
        ref = [rng.gauss(0.001, 0.02) for _ in range(200)]
        detector = DistributionDriftDetector(reference_returns=ref)

        # recent: 큰 mean shift
        for _ in range(60):
            result = detector.update(rng.gauss(-0.05, 0.02))

        assert result.drifted is True
        assert result.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)


class TestCriticalDrift:
    """극심한 분포 이동 → CRITICAL."""

    def test_extreme_shift_critical(self) -> None:
        rng = random.Random(42)
        ref = [rng.gauss(0.001, 0.01) for _ in range(200)]
        detector = DistributionDriftDetector(reference_returns=ref)

        # 완전히 다른 분포
        for _ in range(60):
            result = detector.update(rng.gauss(-0.10, 0.05))

        assert result.severity == DriftSeverity.CRITICAL
        assert result.p_value < 0.01


class TestNaNInfGuard:
    """비정상 입력 무시."""

    def test_nan_ignored(self) -> None:
        detector = DistributionDriftDetector(reference_returns=[0.01] * 100)
        result = detector.update(float("nan"))
        assert result.severity == DriftSeverity.NORMAL
        assert result.recent_n == 0  # NaN은 추가 안됨

    def test_inf_ignored(self) -> None:
        detector = DistributionDriftDetector(reference_returns=[0.01] * 100)
        result = detector.update(float("inf"))
        assert result.severity == DriftSeverity.NORMAL

    def test_neg_inf_ignored(self) -> None:
        detector = DistributionDriftDetector(reference_returns=[0.01] * 100)
        result = detector.update(float("-inf"))
        assert result.severity == DriftSeverity.NORMAL

    def test_reference_nan_filtered(self) -> None:
        """reference에 NaN이 포함되면 필터링."""
        ref = [0.01] * 50 + [float("nan")] * 10
        detector = DistributionDriftDetector(reference_returns=ref)
        assert len(detector._reference_returns) == 50


class TestSerialization:
    """to_dict / restore_from_dict roundtrip."""

    def test_roundtrip(self) -> None:
        detector = DistributionDriftDetector(reference_returns=[0.01] * 100)
        for i in range(40):
            detector.update(0.001 * (i + 1))

        state = detector.to_dict()
        assert "recent_returns" in state
        assert len(state["recent_returns"]) == 40

        # 새 detector에 복원
        detector2 = DistributionDriftDetector(reference_returns=[0.01] * 100)
        detector2.restore_from_dict(state)
        assert len(detector2._recent_returns) == 40
        assert detector2._recent_returns == detector._recent_returns


class TestWindowSize:
    """oldest 제거 동작."""

    def test_window_eviction(self) -> None:
        detector = DistributionDriftDetector(
            reference_returns=[0.01] * 100,
            window_size=40,
        )
        for i in range(50):
            detector.update(0.001 * (i + 1))

        # window_size=40 → 최근 40개만 유지
        assert len(detector._recent_returns) == 40

    def test_oldest_removed(self) -> None:
        detector = DistributionDriftDetector(
            reference_returns=[0.01] * 100,
            window_size=5,
        )
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            detector.update(v)

        assert detector._recent_returns == [2.0, 3.0, 4.0, 5.0, 6.0]


class TestReset:
    """상태 초기화."""

    def test_reset_clears_recent(self) -> None:
        detector = DistributionDriftDetector(reference_returns=[0.01] * 100)
        for _ in range(50):
            detector.update(0.01)
        assert len(detector._recent_returns) == 50

        detector.reset()
        assert len(detector._recent_returns) == 0


class TestResultDataclass:
    """DriftCheckResult 속성 검증."""

    def test_frozen_immutable(self) -> None:
        result = DriftCheckResult(
            severity=DriftSeverity.NORMAL,
            ks_statistic=0.1,
            p_value=0.5,
            recent_n=30,
            drifted=False,
        )
        try:
            result.severity = DriftSeverity.WARNING  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised
