"""Conformal-RANSAC Detector — 구조적 쇠퇴 감지.

누적 수익 시계열에 RANSAC 회귀를 적용하여 robust slope을 추정하고,
Conformal prediction으로 하한을 설정하여 구조적 쇠퇴를 감지합니다.

Severity 판정:
    - slope ≤ 0 → WARNING (양의 alpha 소실)
    - level_breach → WARNING (conformal bound 이탈)
    - both → CRITICAL

Follows GBM Drawdown pattern:
    - __slots__, NaN/Inf guard
    - update(daily_return) -> DecayCheckResult
    - to_dict() / restore_from_dict()

References:
    sklearn.linear_model.RANSACRegressor
    Conformal Prediction (Vovk et al.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DecaySeverity(StrEnum):
    """구조적 쇠퇴 이상 수준."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class DecayCheckResult:
    """Conformal-RANSAC 쇠퇴 체크 결과.

    Attributes:
        severity: 이상 수준
        ransac_slope: RANSAC 추정 기울기
        slope_positive: slope > 0 여부
        conformal_lower_bound: conformal prediction 하한
        current_cumulative: 현재 누적 수익
        level_breach: 하한 이탈 여부
        n_samples: 현재 샘플 수
    """

    severity: DecaySeverity
    ransac_slope: float
    slope_positive: bool
    conformal_lower_bound: float
    current_cumulative: float
    level_breach: bool
    n_samples: int


# ── Constants ─────────────────────────────────────────────────────

_DEFAULT_WINDOW_SIZE = 180
_DEFAULT_ALPHA = 0.05
_DEFAULT_MIN_SAMPLES = 60
_DEFAULT_RANSAC_MIN_RATIO = 0.5


class ConformalRANSACDetector:
    """RANSAC + Conformal Prediction 기반 구조적 쇠퇴 감지기.

    누적 수익 시계열에 RANSAC robust regression을 적합시키고,
    잔차의 conformal prediction으로 하한을 설정합니다.

    Args:
        window_size: 관측 윈도우 크기 (default 180)
        alpha: conformal prediction 유의 수준 (default 0.05)
        min_samples: 판정 시작 최소 샘플 수 (default 60)
        ransac_min_samples_ratio: RANSAC min_samples 비율 (default 0.5)
        ransac_residual_threshold: RANSAC residual threshold (None → auto)
    """

    __slots__ = (
        "_alpha",
        "_cumulative_returns",
        "_daily_returns",
        "_min_samples",
        "_ransac_min_samples_ratio",
        "_ransac_residual_threshold",
        "_window_size",
    )

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        alpha: float = _DEFAULT_ALPHA,
        min_samples: int = _DEFAULT_MIN_SAMPLES,
        ransac_min_samples_ratio: float = _DEFAULT_RANSAC_MIN_RATIO,
        ransac_residual_threshold: float | None = None,
    ) -> None:
        self._window_size = window_size
        self._alpha = alpha
        self._min_samples = min_samples
        self._ransac_min_samples_ratio = ransac_min_samples_ratio
        self._ransac_residual_threshold = ransac_residual_threshold
        self._daily_returns: list[float] = []
        self._cumulative_returns: list[float] = []

    def update(self, daily_return: float) -> DecayCheckResult:
        """새 일별 수익률을 입력하고 구조적 쇠퇴 여부를 판정합니다.

        Args:
            daily_return: 일별 수익률 (e.g. 0.01 = +1%)

        Returns:
            DecayCheckResult
        """
        # NaN/Inf guard
        if not math.isfinite(daily_return):
            return self._normal_result()

        self._daily_returns.append(daily_return)

        # 누적 수익 계산 (기하)
        prev_cum = self._cumulative_returns[-1] if self._cumulative_returns else 0.0
        new_cum = (1.0 + prev_cum) * (1.0 + daily_return) - 1.0
        self._cumulative_returns.append(new_cum)

        # Window eviction
        if len(self._daily_returns) > self._window_size:
            excess = len(self._daily_returns) - self._window_size
            self._daily_returns = self._daily_returns[excess:]
            self._cumulative_returns = self._cumulative_returns[excess:]

        n = len(self._cumulative_returns)

        # 최소 샘플 미달 → NORMAL
        if n < self._min_samples:
            return self._normal_result()

        # RANSAC fit
        x = [float(i) for i in range(n)]
        y = list(self._cumulative_returns)

        try:
            slope, intercept, residuals = self._fit_ransac(x, y)
        except Exception:
            return self._normal_result()

        # RANSAC fitted value at last point
        fitted_last = slope * (n - 1) + intercept

        # Conformal lower bound
        lower_bound = self._conformal_lower_bound(residuals, fitted_last, slope, n)

        # Severity 판정
        slope_positive = slope > 0
        level_breach = y[-1] < lower_bound

        if not slope_positive and level_breach:
            severity = DecaySeverity.CRITICAL
        elif not slope_positive or level_breach:
            severity = DecaySeverity.WARNING
        else:
            severity = DecaySeverity.NORMAL

        return DecayCheckResult(
            severity=severity,
            ransac_slope=slope,
            slope_positive=slope_positive,
            conformal_lower_bound=lower_bound,
            current_cumulative=y[-1],
            level_breach=level_breach,
            n_samples=n,
        )

    def _fit_ransac(
        self,
        x: list[float],
        y: list[float],
    ) -> tuple[float, float, list[float]]:
        """RANSAC regression fit.

        Args:
            x: 시간 인덱스
            y: 누적 수익

        Returns:
            (slope, intercept, residuals) 튜플
        """
        import numpy as np
        from sklearn.linear_model import RANSACRegressor  # type: ignore[import-untyped]

        x_arr = np.array(x, dtype=np.float64).reshape(-1, 1)
        y_arr = np.array(y, dtype=np.float64)

        min_samples = max(2, int(len(x) * self._ransac_min_samples_ratio))

        kwargs: dict[str, Any] = {"min_samples": min_samples, "random_state": 42}
        if self._ransac_residual_threshold is not None:
            kwargs["residual_threshold"] = self._ransac_residual_threshold

        ransac = RANSACRegressor(**kwargs)
        ransac.fit(x_arr, y_arr)

        estimator: Any = ransac.estimator_
        slope = float(estimator.coef_[0])
        intercept = float(estimator.intercept_)

        # 잔차 계산 (inlier만 사용 — conformal bound에 outlier 잔차 혼입 방지)
        y_pred = ransac.predict(x_arr)
        inlier_mask: Any = ransac.inlier_mask_
        all_residuals = y_arr - y_pred
        inlier_residuals = all_residuals[inlier_mask].tolist()

        return slope, intercept, inlier_residuals

    def _conformal_lower_bound(
        self,
        residuals: list[float],
        fitted_last: float,
        slope: float,
        n: int,
    ) -> float:
        """Conformal prediction 하한 계산.

        RANSAC fitted value에서 |residual|의 (1-alpha) quantile을 빼서 하한을 구합니다.

        Args:
            residuals: 잔차 리스트
            fitted_last: 현재 누적 수익 (참조용)
            slope: RANSAC slope
            n: 샘플 수

        Returns:
            conformal lower bound
        """
        abs_residuals = sorted(abs(r) for r in residuals)
        # quantile index: ceil((1-alpha) * n) - 1
        q_idx = min(math.ceil((1 - self._alpha) * len(abs_residuals)) - 1, len(abs_residuals) - 1)
        q_idx = max(q_idx, 0)
        quantile = abs_residuals[q_idx]

        # fitted value at last time point = fitted_last + residual (≈ actual)
        # lower bound = fitted_last - quantile
        return fitted_last - quantile

    def _normal_result(self) -> DecayCheckResult:
        """최소 샘플 미달 시 기본 결과."""
        current = self._cumulative_returns[-1] if self._cumulative_returns else 0.0
        return DecayCheckResult(
            severity=DecaySeverity.NORMAL,
            ransac_slope=0.0,
            slope_positive=True,
            conformal_lower_bound=0.0,
            current_cumulative=current,
            level_breach=False,
            n_samples=len(self._cumulative_returns),
        )

    def reset(self) -> None:
        """Detector 상태 초기화."""
        self._daily_returns = []
        self._cumulative_returns = []

    def to_dict(self) -> dict[str, Any]:
        """Serialize mutable state for persistence."""
        return {
            "daily_returns": list(self._daily_returns),
            "cumulative_returns": list(self._cumulative_returns),
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore mutable state from persisted dict."""
        daily = data.get("daily_returns")
        if isinstance(daily, list):
            self._daily_returns = [float(r) for r in daily if isinstance(r, int | float)]

        cumulative = data.get("cumulative_returns")
        if isinstance(cumulative, list):
            self._cumulative_returns = [float(r) for r in cumulative if isinstance(r, int | float)]
