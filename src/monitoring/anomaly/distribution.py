"""Distribution Drift Detector — KS test 기반 분포 변화 감지.

백테스트 시 관측된 일별 수익률 분포(reference)와 실시간 수익률 분포(recent)를
Kolmogorov-Smirnov 2-sample test로 비교하여 분포 이동을 감지합니다.

Follows GBM Drawdown pattern:
    - Pure Python, __slots__, NaN/Inf guard
    - update(daily_return) -> DriftCheckResult
    - to_dict() / restore_from_dict()

References:
    scipy.stats.ks_2samp — two-sample KS test
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DriftSeverity(StrEnum):
    """Distribution drift 이상 수준."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class DriftCheckResult:
    """KS distribution drift 체크 결과.

    Attributes:
        severity: 이상 수준
        ks_statistic: KS test statistic (0.0 ~ 1.0)
        p_value: KS test p-value
        recent_n: recent sample count
        drifted: 분포 이동 감지 여부
    """

    severity: DriftSeverity
    ks_statistic: float
    p_value: float
    recent_n: int
    drifted: bool


# ── Constants ─────────────────────────────────────────────────────

_MIN_RECENT_SAMPLES = 30
_DEFAULT_WINDOW_SIZE = 60
_DEFAULT_P_WARNING = 0.05
_DEFAULT_P_CRITICAL = 0.01


class DistributionDriftDetector:
    """KS test 기반 Distribution Drift 감지기.

    백테스트 기간의 수익률 분포(reference)를 기준으로,
    최근 rolling window의 수익률 분포가 통계적으로 다른지 검정합니다.

    Args:
        reference_returns: 백테스트 수익률 리스트 (기준 분포)
        window_size: recent rolling window 크기 (default 60)
        p_value_warning: WARNING 임계 p-value (default 0.05)
        p_value_critical: CRITICAL 임계 p-value (default 0.01)
    """

    __slots__ = (
        "_p_value_critical",
        "_p_value_warning",
        "_recent_returns",
        "_reference_returns",
        "_window_size",
    )

    def __init__(
        self,
        reference_returns: list[float],
        window_size: int = _DEFAULT_WINDOW_SIZE,
        p_value_warning: float = _DEFAULT_P_WARNING,
        p_value_critical: float = _DEFAULT_P_CRITICAL,
    ) -> None:
        # NaN/Inf 필터링
        self._reference_returns = [r for r in reference_returns if math.isfinite(r)]
        self._recent_returns: list[float] = []
        self._window_size = window_size
        self._p_value_warning = p_value_warning
        self._p_value_critical = p_value_critical

    def update(self, daily_return: float) -> DriftCheckResult:
        """새 일별 수익률을 입력하고 분포 이동 여부를 판정합니다.

        Args:
            daily_return: 일별 수익률 (e.g. 0.01 = +1%)

        Returns:
            DriftCheckResult
        """
        # NaN/Inf guard
        if not math.isfinite(daily_return):
            return DriftCheckResult(
                severity=DriftSeverity.NORMAL,
                ks_statistic=0.0,
                p_value=1.0,
                recent_n=len(self._recent_returns),
                drifted=False,
            )

        self._recent_returns.append(daily_return)

        # Rolling window: oldest 제거
        if len(self._recent_returns) > self._window_size:
            self._recent_returns = self._recent_returns[-self._window_size :]

        # 최소 샘플 미달 → NORMAL
        if len(self._recent_returns) < _MIN_RECENT_SAMPLES or len(self._reference_returns) < _MIN_RECENT_SAMPLES:
            return DriftCheckResult(
                severity=DriftSeverity.NORMAL,
                ks_statistic=0.0,
                p_value=1.0,
                recent_n=len(self._recent_returns),
                drifted=False,
            )

        # KS 2-sample test
        from scipy.stats import ks_2samp

        stat, p_value = ks_2samp(self._reference_returns, self._recent_returns)

        # Severity 판정
        if p_value < self._p_value_critical:
            severity = DriftSeverity.CRITICAL
        elif p_value < self._p_value_warning:
            severity = DriftSeverity.WARNING
        else:
            severity = DriftSeverity.NORMAL

        drifted = severity != DriftSeverity.NORMAL

        return DriftCheckResult(
            severity=severity,
            ks_statistic=float(stat),
            p_value=float(p_value),
            recent_n=len(self._recent_returns),
            drifted=drifted,
        )

    def reset(self) -> None:
        """Recent returns 초기화."""
        self._recent_returns = []

    def to_dict(self) -> dict[str, Any]:
        """Serialize mutable state for persistence."""
        return {
            "recent_returns": list(self._recent_returns),
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore mutable state from persisted dict."""
        recent = data.get("recent_returns")
        if isinstance(recent, list):
            self._recent_returns = [float(r) for r in recent if isinstance(r, int | float)]
