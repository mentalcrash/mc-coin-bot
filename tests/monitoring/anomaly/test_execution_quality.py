"""Tests for Execution Anomaly Detector."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.monitoring.anomaly.execution_quality import (
    _FILL_RATE_MIN_ORDERS,
    _FILL_RATE_THRESHOLD,
    _LATENCY_COLD_START,
    _REJECTION_CRITICAL_THRESHOLD,
    _REJECTION_WARNING_THRESHOLD,
    _SLIPPAGE_TREND_COUNT,
    ExecutionAnomaly,
    ExecutionAnomalyDetector,
)


class TestLatencyAnomaly:
    """Latency spike 감지 테스트."""

    def test_normal_latency(self) -> None:
        """정상 지연시간 → 이상 없음."""
        detector = ExecutionAnomalyDetector()
        # Build up EWMA with consistent latency
        for _ in range(_LATENCY_COLD_START + 1):
            anomalies = detector.on_fill(0.1, 0.0)
        assert not anomalies

    def test_cold_start_no_anomaly(self) -> None:
        """cold start (< 5 fills) 동안은 anomaly 없음."""
        detector = ExecutionAnomalyDetector()
        for i in range(_LATENCY_COLD_START - 1):
            anomalies = detector.on_fill(10.0 if i > 0 else 0.1, 0.0)
            assert not anomalies

    def test_spike_detected(self) -> None:
        """3x EWMA 초과 시 latency_spike 감지."""
        detector = ExecutionAnomalyDetector()
        # Establish baseline
        for _ in range(10):
            detector.on_fill(0.1, 0.0)
        # Spike
        anomalies = detector.on_fill(5.0, 0.0)
        latency_anomalies = [a for a in anomalies if a.anomaly_type == "latency_spike"]
        assert len(latency_anomalies) == 1
        assert latency_anomalies[0].severity == "WARNING"

    def test_nan_latency_ignored(self) -> None:
        """NaN latency는 무시."""
        detector = ExecutionAnomalyDetector()
        for _ in range(10):
            detector.on_fill(0.1, 0.0)
        anomalies = detector.on_fill(float("nan"), 0.0)
        assert not any(a.anomaly_type == "latency_spike" for a in anomalies)


class TestConsecutiveRejections:
    """연속 거부 감지 테스트."""

    def test_below_threshold_no_anomaly(self) -> None:
        """임계값 미만 → 이상 없음."""
        detector = ExecutionAnomalyDetector()
        for _ in range(_REJECTION_WARNING_THRESHOLD - 1):
            anomalies = detector.on_rejection()
        assert not anomalies

    def test_warning_at_threshold(self) -> None:
        """3건 연속 → WARNING."""
        detector = ExecutionAnomalyDetector()
        anomalies: list[ExecutionAnomaly] = []
        for _ in range(_REJECTION_WARNING_THRESHOLD):
            anomalies = detector.on_rejection()
        assert len(anomalies) == 1
        assert anomalies[0].severity == "WARNING"
        assert anomalies[0].anomaly_type == "consecutive_rejections"

    def test_critical_at_threshold(self) -> None:
        """5건 연속 → CRITICAL."""
        detector = ExecutionAnomalyDetector()
        anomalies: list[ExecutionAnomaly] = []
        for _ in range(_REJECTION_CRITICAL_THRESHOLD):
            anomalies = detector.on_rejection()
        assert len(anomalies) == 1
        assert anomalies[0].severity == "CRITICAL"

    def test_fill_resets_rejection_count(self) -> None:
        """fill이 rejection count를 리셋."""
        detector = ExecutionAnomalyDetector()
        # 2 rejections
        for _ in range(_REJECTION_WARNING_THRESHOLD - 1):
            detector.on_rejection()
        # Fill → reset
        detector.on_fill(0.1, 0.0)
        # 1 more rejection → still below threshold
        anomalies = detector.on_rejection()
        assert not anomalies


class TestFillRate:
    """Fill rate 감지 테스트."""

    def test_insufficient_orders(self) -> None:
        """주문 수 부족 시 None."""
        detector = ExecutionAnomalyDetector()
        for _ in range(_FILL_RATE_MIN_ORDERS - 1):
            detector.on_order_request()
        result = detector.check_fill_rate()
        assert result is None

    def test_high_fill_rate_normal(self) -> None:
        """정상 fill rate → None."""
        detector = ExecutionAnomalyDetector()
        for _ in range(10):
            detector.on_order_request()
            detector.on_fill(0.1, 0.0)
        result = detector.check_fill_rate()
        assert result is None

    def test_low_fill_rate_warning(self) -> None:
        """낮은 fill rate → WARNING."""
        detector = ExecutionAnomalyDetector()
        for _ in range(10):
            detector.on_order_request()
        # Only 2 fills out of 10
        detector.on_fill(0.1, 0.0)
        detector.on_fill(0.1, 0.0)
        result = detector.check_fill_rate()
        assert result is not None
        assert result.anomaly_type == "low_fill_rate"
        assert result.severity == "WARNING"
        assert result.current_value < _FILL_RATE_THRESHOLD


class TestSlippageTrend:
    """Slippage trend 감지 테스트."""

    def test_no_trend(self) -> None:
        """랜덤 슬리피지 → 이상 없음."""
        detector = ExecutionAnomalyDetector()
        for _ in range(5):
            detector.on_fill(0.1, 0.0)  # Need cold start fills
        anomalies = detector.on_fill(0.1, 5.0)
        anomalies = detector.on_fill(0.1, 3.0)  # Decrease → reset
        anomalies = detector.on_fill(0.1, 4.0)
        assert not any(a.anomaly_type == "slippage_trend" for a in anomalies)

    def test_increasing_trend_detected(self) -> None:
        """연속 증가 → WARNING."""
        detector = ExecutionAnomalyDetector()
        # Build up enough fills to pass cold start
        for _ in range(_LATENCY_COLD_START + 1):
            detector.on_fill(0.1, 0.0)

        # _SLIPPAGE_TREND_COUNT + 1 increasing values
        detector.on_fill(0.1, 1.0)  # base
        results: list[ExecutionAnomaly] = []
        for i in range(_SLIPPAGE_TREND_COUNT):
            anomalies = detector.on_fill(0.1, 2.0 + i)
            results.extend(a for a in anomalies if a.anomaly_type == "slippage_trend")

        assert len(results) >= 1
        assert results[0].severity == "WARNING"

    def test_nan_slippage_ignored(self) -> None:
        """NaN slippage는 무시."""
        detector = ExecutionAnomalyDetector()
        for _ in range(10):
            detector.on_fill(0.1, 0.0)
        anomalies = detector.on_fill(0.1, float("nan"))
        assert not any(a.anomaly_type == "slippage_trend" for a in anomalies)


class TestSerialization:
    """to_dict / restore_from_dict 테스트."""

    def test_roundtrip(self) -> None:
        detector = ExecutionAnomalyDetector()
        for _ in range(5):
            detector.on_fill(0.1, 1.0)
        detector.on_rejection()
        detector.on_rejection()

        data = detector.to_dict()
        restored = ExecutionAnomalyDetector()
        restored.restore_from_dict(data)

        assert restored.to_dict()["avg_latency"] == pytest.approx(data["avg_latency"])
        assert restored.to_dict()["latency_count"] == data["latency_count"]
        assert restored.to_dict()["consecutive_rejections"] == data["consecutive_rejections"]

    def test_empty_restore(self) -> None:
        """빈 dict 복원 → 기본값."""
        detector = ExecutionAnomalyDetector()
        detector.restore_from_dict({})
        assert detector.to_dict()["avg_latency"] == 0.0
        assert detector.to_dict()["latency_count"] == 0


class TestEdgeCases:
    """엣지 케이스 테스트."""

    def test_negative_latency_ignored(self) -> None:
        """음수 latency는 무시."""
        detector = ExecutionAnomalyDetector()
        for _ in range(10):
            detector.on_fill(0.1, 0.0)
        anomalies = detector.on_fill(-1.0, 0.0)
        assert not any(a.anomaly_type == "latency_spike" for a in anomalies)

    def test_expired_timestamps_pruned(self) -> None:
        """1h 이상 된 타임스탬프는 제거."""
        detector = ExecutionAnomalyDetector()

        # Use monotonic mock to simulate time passing
        base_time = time.monotonic()

        with patch("src.monitoring.anomaly.execution_quality.time.monotonic") as mock_time:
            # Orders at t=0
            mock_time.return_value = base_time
            for _ in range(10):
                detector.on_order_request()
                detector.on_fill(0.1, 0.0)

            # Check fill rate at t=0: 100%
            result = detector.check_fill_rate()
            assert result is None

            # Jump 2 hours ahead — all old timestamps should be pruned
            mock_time.return_value = base_time + 7200
            result = detector.check_fill_rate()
            # After pruning, order count < MIN_ORDERS → None
            assert result is None
