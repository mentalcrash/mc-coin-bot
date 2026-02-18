"""Execution Anomaly Detector — 실행 품질 이상 패턴 streaming 감지.

4가지 이상 패턴:
1. Latency spike: EWMA(alpha=0.1) 대비 3x 초과
2. Consecutive rejections: 연속 3건 WARNING, 5건 CRITICAL
3. Low fill rate: 1h sliding window, < 80% (min 5건)
4. Slippage trend: 연속 3회 증가 WARNING

Follows PageHinkley pattern: Pure Python, __slots__, NaN/Inf guard.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ExecutionAnomaly:
    """실행 이상 감지 결과.

    Attributes:
        anomaly_type: 이상 유형
        severity: WARNING or CRITICAL
        message: 상세 메시지
        current_value: 현재 측정값
        threshold: 임계값
    """

    anomaly_type: str  # latency_spike | consecutive_rejections | low_fill_rate | slippage_trend
    severity: Literal["WARNING", "CRITICAL"]
    message: str
    current_value: float
    threshold: float


# ── Constants ─────────────────────────────────────────────────────

_EWMA_ALPHA = 0.1
_LATENCY_SPIKE_MULTIPLIER = 3.0
_LATENCY_COLD_START = 5

_REJECTION_WARNING_THRESHOLD = 3
_REJECTION_CRITICAL_THRESHOLD = 5

_FILL_RATE_WINDOW_SECONDS = 3600.0  # 1h
_FILL_RATE_THRESHOLD = 0.80
_FILL_RATE_MIN_ORDERS = 5

_SLIPPAGE_TREND_COUNT = 3


class ExecutionAnomalyDetector:
    """실행 품질 이상 감지기.

    on_fill() / on_order_request() / on_rejection() 으로
    streaming 방식으로 이상 패턴을 감지합니다.
    """

    __slots__ = (
        "_avg_latency",
        "_consecutive_rejections",
        "_fill_timestamps",
        "_last_slippages",
        "_latency_count",
        "_order_timestamps",
        "_slippage_increase_count",
    )

    def __init__(self) -> None:
        self._avg_latency: float = 0.0
        self._latency_count: int = 0
        self._consecutive_rejections: int = 0

        # 1h sliding window timestamps
        self._order_timestamps: list[float] = []
        self._fill_timestamps: list[float] = []

        # Slippage trend tracking
        self._last_slippages: list[float] = []
        self._slippage_increase_count: int = 0

    @property
    def consecutive_rejections(self) -> int:
        """현재 연속 거부 횟수."""
        return self._consecutive_rejections

    def get_fill_rate(self) -> float:
        """현재 1h 윈도우 fill rate (0.0~1.0).

        Returns:
            Fill rate. 주문 없으면 1.0 (정상 취급).
        """
        self._prune_timestamps()
        order_count = len(self._order_timestamps)
        if order_count == 0:
            return 1.0
        return len(self._fill_timestamps) / order_count

    def on_fill(
        self,
        latency_seconds: float,
        slippage_bps: float,
    ) -> list[ExecutionAnomaly]:
        """체결 이벤트 처리 → 이상 감지.

        Args:
            latency_seconds: 주문→체결 지연시간 (초)
            slippage_bps: 슬리피지 (basis points)

        Returns:
            감지된 이상 리스트 (빈 리스트 = 정상)
        """
        anomalies: list[ExecutionAnomaly] = []

        # Reset consecutive rejections on successful fill
        self._consecutive_rejections = 0

        # Record fill timestamp
        now = time.monotonic()
        self._fill_timestamps.append(now)
        self._prune_timestamps()

        # 1. Latency spike detection
        latency_anomaly = self._check_latency(latency_seconds)
        if latency_anomaly is not None:
            anomalies.append(latency_anomaly)

        # 2. Slippage trend detection
        slippage_anomaly = self._check_slippage_trend(slippage_bps)
        if slippage_anomaly is not None:
            anomalies.append(slippage_anomaly)

        return anomalies

    def on_order_request(self) -> None:
        """주문 요청 이벤트 기록."""
        now = time.monotonic()
        self._order_timestamps.append(now)
        self._prune_timestamps()

    def on_rejection(self) -> list[ExecutionAnomaly]:
        """주문 거부 이벤트 처리 → 연속 거부 감지.

        Returns:
            감지된 이상 리스트
        """
        anomalies: list[ExecutionAnomaly] = []

        self._consecutive_rejections += 1
        count = self._consecutive_rejections

        if count >= _REJECTION_CRITICAL_THRESHOLD:
            anomalies.append(
                ExecutionAnomaly(
                    anomaly_type="consecutive_rejections",
                    severity="CRITICAL",
                    message=f"Consecutive rejections: {count} (critical >= {_REJECTION_CRITICAL_THRESHOLD})",
                    current_value=float(count),
                    threshold=float(_REJECTION_CRITICAL_THRESHOLD),
                )
            )
        elif count >= _REJECTION_WARNING_THRESHOLD:
            anomalies.append(
                ExecutionAnomaly(
                    anomaly_type="consecutive_rejections",
                    severity="WARNING",
                    message=f"Consecutive rejections: {count} (warn >= {_REJECTION_WARNING_THRESHOLD})",
                    current_value=float(count),
                    threshold=float(_REJECTION_WARNING_THRESHOLD),
                )
            )

        return anomalies

    def check_fill_rate(self) -> ExecutionAnomaly | None:
        """Fill rate 주기적 체크 (외부 호출용).

        Returns:
            Fill rate 이상 또는 None (정상/데이터 부족)
        """
        self._prune_timestamps()

        order_count = len(self._order_timestamps)
        if order_count < _FILL_RATE_MIN_ORDERS:
            return None

        fill_count = len(self._fill_timestamps)
        fill_rate = fill_count / order_count

        if fill_rate < _FILL_RATE_THRESHOLD:
            return ExecutionAnomaly(
                anomaly_type="low_fill_rate",
                severity="WARNING",
                message=(
                    f"Low fill rate: {fill_rate:.1%} "
                    f"({fill_count}/{order_count} in 1h, threshold {_FILL_RATE_THRESHOLD:.0%})"
                ),
                current_value=fill_rate,
                threshold=_FILL_RATE_THRESHOLD,
            )

        return None

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize mutable state for persistence."""
        return {
            "avg_latency": self._avg_latency,
            "latency_count": self._latency_count,
            "consecutive_rejections": self._consecutive_rejections,
            "slippage_increase_count": self._slippage_increase_count,
            "last_slippages": list(self._last_slippages),
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore mutable state from persisted dict.

        Note: _order_timestamps / _fill_timestamps are monotonic clock-based
        and meaningless across process restarts, so they are explicitly cleared.
        """
        self._avg_latency = float(data.get("avg_latency", 0.0))
        self._latency_count = int(data.get("latency_count", 0))
        self._consecutive_rejections = int(data.get("consecutive_rejections", 0))
        self._slippage_increase_count = int(data.get("slippage_increase_count", 0))
        raw = data.get("last_slippages", [])
        self._last_slippages = [float(x) for x in raw] if isinstance(raw, list) else []
        # Monotonic timestamps are invalid after restart
        self._order_timestamps = []
        self._fill_timestamps = []

    # ── Private ──────────────────────────────────────────────────

    def _check_latency(self, latency: float) -> ExecutionAnomaly | None:
        """EWMA 기반 latency spike 감지."""
        if math.isnan(latency) or math.isinf(latency) or latency < 0:
            return None

        self._latency_count += 1

        if self._latency_count == 1:
            self._avg_latency = latency
            return None

        # EWMA update
        self._avg_latency = (
            _EWMA_ALPHA * latency + (1.0 - _EWMA_ALPHA) * self._avg_latency
        )

        # Cold start: skip anomaly detection
        if self._latency_count < _LATENCY_COLD_START:
            return None

        threshold = self._avg_latency * _LATENCY_SPIKE_MULTIPLIER
        if threshold > 0 and latency > threshold:
            return ExecutionAnomaly(
                anomaly_type="latency_spike",
                severity="WARNING",
                message=(
                    f"Latency spike: {latency:.2f}s > {_LATENCY_SPIKE_MULTIPLIER:.0f}x avg ({self._avg_latency:.2f}s)"
                ),
                current_value=latency,
                threshold=threshold,
            )

        return None

    def _check_slippage_trend(self, bps: float) -> ExecutionAnomaly | None:
        """연속 증가하는 슬리피지 트렌드 감지."""
        if math.isnan(bps) or math.isinf(bps):
            return None

        if self._last_slippages and bps > self._last_slippages[-1]:
            self._slippage_increase_count += 1
        else:
            self._slippage_increase_count = 0

        self._last_slippages.append(bps)
        # Keep only recent slippages for memory efficiency
        if len(self._last_slippages) > _SLIPPAGE_TREND_COUNT + 1:
            self._last_slippages = self._last_slippages[-(
                _SLIPPAGE_TREND_COUNT + 1
            ) :]

        if self._slippage_increase_count >= _SLIPPAGE_TREND_COUNT:
            return ExecutionAnomaly(
                anomaly_type="slippage_trend",
                severity="WARNING",
                message=(
                    f"Slippage increasing: {self._slippage_increase_count} consecutive increases, "
                    f"latest {bps:.1f}bps"
                ),
                current_value=bps,
                threshold=float(_SLIPPAGE_TREND_COUNT),
            )

        return None

    def _prune_timestamps(self) -> None:
        """1h sliding window 밖의 타임스탬프를 제거."""
        now = time.monotonic()
        cutoff = now - _FILL_RATE_WINDOW_SECONDS
        self._order_timestamps = [t for t in self._order_timestamps if t >= cutoff]
        self._fill_timestamps = [t for t in self._fill_timestamps if t >= cutoff]
