"""Tests for Conformal-RANSAC Detector."""

from __future__ import annotations

import math
import random

import pytest

from src.monitoring.anomaly.conformal_ransac import (
    ConformalRANSACDetector,
    DecayCheckResult,
    DecaySeverity,
)


class TestInitialState:
    """최소 샘플 미달 시 NORMAL."""

    def test_empty_returns_normal(self) -> None:
        detector = ConformalRANSACDetector(min_samples=60)
        result = detector.update(0.01)
        assert result.severity == DecaySeverity.NORMAL
        assert result.n_samples == 1

    def test_under_min_samples_normal(self) -> None:
        """59개 → NORMAL."""
        detector = ConformalRANSACDetector(min_samples=60)
        for _ in range(59):
            result = detector.update(0.01)
        assert result.severity == DecaySeverity.NORMAL
        assert result.n_samples == 59


class TestPositiveTrend:
    """안정적 양의 수익 → NORMAL."""

    def test_stable_positive_returns(self) -> None:
        detector = ConformalRANSACDetector(min_samples=30, window_size=100)
        rng = random.Random(42)
        for _ in range(80):
            result = detector.update(0.005 + rng.gauss(0, 0.001))
        assert result.severity == DecaySeverity.NORMAL
        assert result.slope_positive is True


class TestSlopeDecay:
    """Drift 소실 → slope ≤ 0 → WARNING."""

    def test_decaying_returns_warning(self) -> None:
        detector = ConformalRANSACDetector(min_samples=30, window_size=120)
        # 초반 양의 수익
        for _ in range(40):
            detector.update(0.01)
        # 이후 지속적 음의 수익 → slope 하락
        for _ in range(80):
            result = detector.update(-0.005)

        # slope이 0 이하이거나 level breach 감지
        assert result.severity in (DecaySeverity.WARNING, DecaySeverity.CRITICAL)


class TestLevelBreach:
    """Conformal bound 이탈 → WARNING."""

    def test_sudden_crash_breach(self) -> None:
        detector = ConformalRANSACDetector(min_samples=30, window_size=100)
        # 안정적 양의 수익
        for _ in range(60):
            detector.update(0.005)
        # 급락
        for _ in range(20):
            result = detector.update(-0.05)

        # level breach 또는 slope decay 감지
        assert result.severity in (DecaySeverity.WARNING, DecaySeverity.CRITICAL)


class TestCriticalBothConditions:
    """slope ≤ 0 AND level breach → CRITICAL."""

    def test_both_conditions_critical(self) -> None:
        detector = ConformalRANSACDetector(min_samples=30, window_size=150)
        # 초반 약한 양의 수익
        for _ in range(40):
            detector.update(0.002)
        # 장기 하락 → slope 음전환 + level breach
        for _ in range(110):
            result = detector.update(-0.01)

        assert result.severity == DecaySeverity.CRITICAL
        assert result.slope_positive is False
        assert result.level_breach is True


class TestOutlierRobustness:
    """단일 outlier → false alarm 미발생."""

    def test_single_outlier_no_alarm(self) -> None:
        detector = ConformalRANSACDetector(min_samples=30, window_size=100)
        rng = random.Random(42)
        for _ in range(70):
            detector.update(0.005 + rng.gauss(0, 0.002))
        # 단일 extreme outlier
        detector.update(-0.30)
        # 복귀
        for _ in range(10):
            result = detector.update(0.005 + rng.gauss(0, 0.002))

        # RANSAC은 outlier에 robust → slope 유지
        assert result.slope_positive is True


class TestNaNInfGuard:
    """비정상 입력 무시."""

    def test_nan_ignored(self) -> None:
        detector = ConformalRANSACDetector(min_samples=30)
        result = detector.update(float("nan"))
        assert result.severity == DecaySeverity.NORMAL
        assert result.n_samples == 0

    def test_inf_ignored(self) -> None:
        detector = ConformalRANSACDetector(min_samples=30)
        result = detector.update(float("inf"))
        assert result.severity == DecaySeverity.NORMAL

    def test_mixed_nan_in_stream(self) -> None:
        """NaN이 중간에 섞여도 유효 데이터만 사용."""
        detector = ConformalRANSACDetector(min_samples=30, window_size=100)
        for _ in range(35):
            detector.update(0.01)
        detector.update(float("nan"))
        result = detector.update(0.01)
        # NaN은 무시, 총 36개 유효 샘플
        assert result.n_samples == 36


class TestSerialization:
    """to_dict / restore_from_dict roundtrip."""

    def test_roundtrip(self) -> None:
        detector = ConformalRANSACDetector(min_samples=30, window_size=100)
        for i in range(50):
            detector.update(0.001 * (i + 1))

        state = detector.to_dict()
        assert "daily_returns" in state
        assert "cumulative_returns" in state
        assert len(state["daily_returns"]) == 50

        # 새 detector에 복원
        detector2 = ConformalRANSACDetector(min_samples=30, window_size=100)
        detector2.restore_from_dict(state)
        assert len(detector2._daily_returns) == 50
        assert len(detector2._cumulative_returns) == 50


class TestWindowEviction:
    """window_size 초과 시 oldest 제거."""

    def test_window_eviction(self) -> None:
        detector = ConformalRANSACDetector(min_samples=10, window_size=30)
        for i in range(50):
            detector.update(0.001)

        assert len(detector._daily_returns) == 30
        assert len(detector._cumulative_returns) == 30


class TestReset:
    """상태 초기화."""

    def test_reset_clears_state(self) -> None:
        detector = ConformalRANSACDetector(min_samples=30)
        for _ in range(50):
            detector.update(0.01)
        assert len(detector._daily_returns) == 50

        detector.reset()
        assert len(detector._daily_returns) == 0
        assert len(detector._cumulative_returns) == 0


class TestResultDataclass:
    """DecayCheckResult 속성 검증."""

    def test_frozen_immutable(self) -> None:
        result = DecayCheckResult(
            severity=DecaySeverity.NORMAL,
            ransac_slope=0.01,
            slope_positive=True,
            conformal_lower_bound=0.0,
            current_cumulative=0.1,
            level_breach=False,
            n_samples=60,
        )
        with pytest.raises(AttributeError):
            result.severity = DecaySeverity.WARNING  # type: ignore[misc]
