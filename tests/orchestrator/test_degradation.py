"""Tests for PageHinkleyDetector — CUSUM variant mean-shift detector."""

from __future__ import annotations

import pytest

from src.orchestrator.degradation import PageHinkleyDetector

# ── TestInitialState ─────────────────────────────────────────────


class TestInitialState:
    def test_initial_score_zero(self) -> None:
        det = PageHinkleyDetector()
        assert det.score == pytest.approx(0.0)

    def test_initial_n_observations_zero(self) -> None:
        det = PageHinkleyDetector()
        assert det.n_observations == 0

    def test_single_observation_no_detection(self) -> None:
        """첫 관측값은 x_mean 초기화만 → 감지 불가."""
        det = PageHinkleyDetector()
        result = det.update(0.01)
        assert result is False
        assert det.n_observations == 1

    def test_two_observations_no_detection(self) -> None:
        """소량 데이터로는 감지 불가."""
        det = PageHinkleyDetector()
        det.update(0.01)
        result = det.update(0.01)
        assert result is False
        assert det.n_observations == 2


# ── TestNoDetection ──────────────────────────────────────────────


class TestNoDetection:
    def test_constant_values_no_detection(self) -> None:
        """일정한 값 → 감지 없음."""
        det = PageHinkleyDetector()
        for _ in range(200):
            result = det.update(0.01)
            assert result is False

    def test_stable_positive_returns_no_detection(self) -> None:
        """안정적 양수 수익률 → 감지 없음."""
        det = PageHinkleyDetector()
        for _ in range(200):
            result = det.update(0.005)
            assert result is False

    def test_alternating_signs_no_detection(self) -> None:
        """부호 교대 (mean ≈ 0) → 감지 없음."""
        det = PageHinkleyDetector()
        for i in range(200):
            value = 0.01 if i % 2 == 0 else -0.01
            result = det.update(value)
            assert result is False


# ── TestDetection ────────────────────────────────────────────────


class TestDetection:
    def test_mean_shift_detected(self) -> None:
        """양수 → 음수 mean shift 감지."""
        det = PageHinkleyDetector(delta=0.005, lambda_=50.0, alpha=0.9999)
        # 정상 기간
        for _ in range(100):
            det.update(0.01)

        # Mean shift: 큰 음수 값 지속 (alpha=0.9999 slow-adapting baseline)
        detected = False
        for _ in range(1500):
            if det.update(-0.05):
                detected = True
                break

        assert detected, "Mean shift should be detected"

    def test_gradual_drift_detected(self) -> None:
        """점진적 하락도 결국 감지."""
        det = PageHinkleyDetector(delta=0.001, lambda_=20.0, alpha=0.9999)
        for _ in range(50):
            det.update(0.01)

        detected = False
        for i in range(500):
            # 점진적으로 악화
            value = 0.01 - i * 0.001
            if det.update(value):
                detected = True
                break

        assert detected, "Gradual drift should be detected"

    def test_score_increases_under_shift(self) -> None:
        """음수 shift 시 score 단조 증가."""
        det = PageHinkleyDetector(delta=0.005, lambda_=50.0, alpha=0.9999)
        for _ in range(50):
            det.update(0.01)

        # 음수 shift 시작 후 score 추적
        scores: list[float] = []
        for _ in range(50):
            det.update(-0.05)
            scores.append(det.score)

        # Score가 증가 추세 (매 step마다 반드시 증가하진 않지만 전반적 증가)
        assert scores[-1] > scores[0], "Score should increase under negative shift"

    def test_large_sudden_drop_detected(self) -> None:
        """큰 급락 감지."""
        det = PageHinkleyDetector(delta=0.001, lambda_=10.0, alpha=0.9999)
        for _ in range(50):
            det.update(0.01)

        detected = False
        for _ in range(200):
            if det.update(-0.5):
                detected = True
                break

        assert detected, "Large sudden drop should be detected"


# ── TestReset ────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self) -> None:
        """reset() 후 초기 상태 복원."""
        det = PageHinkleyDetector()
        for _ in range(50):
            det.update(0.01)

        det.reset()
        assert det.score == pytest.approx(0.0)
        assert det.n_observations == 0

    def test_no_detection_after_reset(self) -> None:
        """reset 후 이전 이력 무효화."""
        det = PageHinkleyDetector(delta=0.005, lambda_=50.0)
        # 열화 직전까지 진행
        for _ in range(50):
            det.update(0.01)
        for _ in range(30):
            det.update(-0.05)

        score_before = det.score
        assert score_before > 0

        det.reset()

        # Reset 후 정상 데이터 → 감지 없음
        for _ in range(50):
            result = det.update(0.01)
            assert result is False


# ── TestParameterSensitivity ─────────────────────────────────────


class TestParameterSensitivity:
    def test_higher_lambda_fewer_detections(self) -> None:
        """lambda 높을수록 감지 어려움 (더 많은 데이터 필요)."""
        steps_low = self._steps_to_detect(lambda_=10.0)
        steps_high = self._steps_to_detect(lambda_=100.0)
        assert steps_high > steps_low

    def test_higher_delta_fewer_detections(self) -> None:
        """delta 높을수록 작은 변화 무시."""
        steps_low = self._steps_to_detect(delta=0.001)
        steps_high = self._steps_to_detect(delta=0.01)
        assert steps_high > steps_low

    @staticmethod
    def _steps_to_detect(
        delta: float = 0.005,
        lambda_: float = 50.0,
    ) -> int:
        """음수 shift 후 감지까지 step 수."""
        det = PageHinkleyDetector(delta=delta, lambda_=lambda_, alpha=0.9999)
        for _ in range(50):
            det.update(0.01)

        for step in range(1, 2000):
            if det.update(-0.05):
                return step

        return 2000  # Not detected within limit


# ── TestEdgeCases ────────────────────────────────────────────────


class TestNaNInfGuard:
    """H-7: NaN/Inf 입력 시 상태 미오염 검증."""

    def test_nan_returns_false(self) -> None:
        """NaN → False, 상태 미오염."""
        det = PageHinkleyDetector()
        det.update(0.01)
        n_before = det.n_observations
        result = det.update(float("nan"))
        assert result is False
        assert det.n_observations == n_before

    def test_inf_returns_false(self) -> None:
        """Inf → False, 상태 미오염."""
        det = PageHinkleyDetector()
        det.update(0.01)
        n_before = det.n_observations
        score_before = det.score
        result = det.update(float("inf"))
        assert result is False
        assert det.n_observations == n_before
        assert det.score == pytest.approx(score_before)

    def test_neg_inf_returns_false(self) -> None:
        """-Inf → False, 상태 미오염."""
        det = PageHinkleyDetector()
        det.update(0.01)
        result = det.update(float("-inf"))
        assert result is False
        assert det.n_observations == 1

    def test_nan_then_normal_works(self) -> None:
        """NaN 후 정상값 → 정상 동작."""
        det = PageHinkleyDetector()
        det.update(float("nan"))
        assert det.n_observations == 0
        det.update(0.01)
        assert det.n_observations == 1
        det.update(0.02)
        assert det.n_observations == 2

    def test_first_observation_nan_skip(self) -> None:
        """첫 관측값 NaN → skip, 다음 값으로 초기화."""
        det = PageHinkleyDetector()
        det.update(float("nan"))
        assert det.n_observations == 0
        det.update(0.05)
        assert det.n_observations == 1
        # 두번째 값이 첫 관측으로 초기화됨
        assert det.score == pytest.approx(0.0)


class TestM6DefaultAlpha:
    """M-6: 기본 alpha=0.99 → 반감기 ~69일, 빠른 적응."""

    def test_default_alpha_is_099(self) -> None:
        """기본 alpha가 0.99로 설정됨."""
        det = PageHinkleyDetector()
        assert det._alpha == pytest.approx(0.99)

    def test_alpha_099_adapts_faster(self) -> None:
        """alpha=0.99는 0.9999보다 x_mean이 빠르게 적응."""
        det_fast = PageHinkleyDetector(alpha=0.99)
        det_slow = PageHinkleyDetector(alpha=0.9999)

        # 동일 데이터 입력
        for _ in range(50):
            det_fast.update(0.10)
            det_slow.update(0.10)

        # 이후 방향 전환
        for _ in range(50):
            det_fast.update(-0.10)
            det_slow.update(-0.10)

        # alpha=0.99: x_mean이 빠르게 -0.10쪽으로 이동 → score 낮음
        # alpha=0.9999: x_mean이 느리게 이동 → score 높음
        assert det_fast.score < det_slow.score


class TestEdgeCases:
    def test_zero_variance_no_detection(self) -> None:
        """모든 값이 0 → 감지 없음."""
        det = PageHinkleyDetector()
        for _ in range(200):
            result = det.update(0.0)
        # delta > 0이므로 m_t는 음수로 감소하고, score는 0에 유지
        assert result is False

    def test_slots_memory_efficiency(self) -> None:
        """__slots__ 사용 확인."""
        det = PageHinkleyDetector()
        assert not hasattr(det, "__dict__")
