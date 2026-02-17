"""GBM Drawdown Monitor — 통계적 drawdown 이상 감지.

Geometric Brownian Motion 가정 하에 expected maximum drawdown(depth/duration)을
Magdon-Ismail 근사로 계산하고, 실제 drawdown이 통계적으로 비정상인지 판정합니다.

Follows PageHinkley pattern:
    - Pure Python, __slots__, NaN/Inf guard
    - update(daily_return) -> DrawdownCheckResult
    - to_dict() / restore_from_dict()

References:
    Magdon-Ismail, M., Atiya, A. F., Pratap, A., & Abu-Mostafa, Y. S. (2004).
    "On the Maximum Drawdown of a Brownian Motion."
    Journal of Applied Probability, 41(1), 147-161.
    DOI: 10.1239/jap/1077134674
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DrawdownSeverity(StrEnum):
    """Drawdown 이상 수준."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class DrawdownCheckResult:
    """GBM drawdown 체크 결과.

    Attributes:
        severity: 이상 수준
        current_depth: 현재 drawdown 비율 (0.0 ~ 1.0)
        current_duration_days: 현재 drawdown 지속 일수
        expected_max_depth: GBM 기반 예상 최대 drawdown
        expected_max_duration: GBM 기반 예상 최대 drawdown 지속 일수
        depth_exceeded: depth가 expected를 초과했는지
        duration_exceeded: duration이 expected를 초과했는지
    """

    severity: DrawdownSeverity
    current_depth: float
    current_duration_days: int
    expected_max_depth: float
    expected_max_duration: int
    depth_exceeded: bool
    duration_exceeded: bool


# ── Constants ─────────────────────────────────────────────────────

_MIN_OBSERVATION_DAYS = 30
_EPSILON = 1e-12

# Rational approximation coefficients for norm_ppf (Abramowitz & Stegun 26.2.23)
_PPF_A0 = 2.515517
_PPF_A1 = 0.802853
_PPF_A2 = 0.010328
_PPF_B1 = 1.432788
_PPF_B2 = 0.189269
_PPF_B3 = 0.001308


class GBMDrawdownMonitor:
    """GBM 기반 Drawdown 이상 감지기.

    Daily return 스트림을 수신하여 peak equity 대비 drawdown을
    추적하고, GBM 모형에서 기대되는 최대 drawdown(depth/duration)과 비교합니다.

    Args:
        mu: 일별 기대 수익률 (drift)
        sigma: 일별 변동성
        confidence: 신뢰 수준 (0.0~1.0). Default 0.95 → 95% VaR.
    """

    __slots__ = (
        "_confidence",
        "_current_equity",
        "_dd_start_day",
        "_in_drawdown",
        "_mu",
        "_n_days",
        "_peak_equity",
        "_sigma",
        "_z_score",
    )

    def __init__(
        self,
        mu: float,
        sigma: float,
        confidence: float = 0.95,
    ) -> None:
        self._mu = mu
        self._sigma = max(sigma, _EPSILON)  # sigma=0 guard
        self._confidence = confidence
        self._z_score = _norm_ppf(confidence)

        self._peak_equity: float = 1.0
        self._current_equity: float = 1.0
        self._dd_start_day: int = 0
        self._n_days: int = 0
        self._in_drawdown: bool = False

    def update(self, daily_return: float) -> DrawdownCheckResult:
        """새 일별 수익률을 입력하고 drawdown 이상 여부를 판정합니다.

        Args:
            daily_return: 일별 수익률 (e.g. 0.01 = +1%)

        Returns:
            DrawdownCheckResult
        """
        # NaN/Inf guard
        if math.isnan(daily_return) or math.isinf(daily_return):
            return DrawdownCheckResult(
                severity=DrawdownSeverity.NORMAL,
                current_depth=0.0,
                current_duration_days=0,
                expected_max_depth=0.0,
                expected_max_duration=0,
                depth_exceeded=False,
                duration_exceeded=False,
            )

        self._n_days += 1
        self._current_equity *= 1.0 + daily_return

        # Peak 갱신
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity
            self._in_drawdown = False
            self._dd_start_day = self._n_days

        # Current depth
        depth = 0.0
        if self._peak_equity > _EPSILON:
            depth = (self._peak_equity - self._current_equity) / self._peak_equity
        depth = max(depth, 0.0)

        # Drawdown duration
        if depth > _EPSILON:
            if not self._in_drawdown:
                self._in_drawdown = True
                self._dd_start_day = self._n_days
            duration = self._n_days - self._dd_start_day + 1
        else:
            self._in_drawdown = False
            duration = 0

        # n < MIN_OBSERVATION_DAYS → 통계적 의미 부족, 항상 NORMAL
        if self._n_days < _MIN_OBSERVATION_DAYS:
            return DrawdownCheckResult(
                severity=DrawdownSeverity.NORMAL,
                current_depth=depth,
                current_duration_days=duration,
                expected_max_depth=0.0,
                expected_max_duration=0,
                depth_exceeded=False,
                duration_exceeded=False,
            )

        # Expected max drawdown / duration
        exp_depth = self.expected_max_drawdown(self._n_days)
        exp_duration = self.expected_max_duration(self._n_days)

        depth_exceeded = depth > exp_depth
        duration_exceeded = duration > exp_duration

        # Severity
        if depth_exceeded and duration_exceeded:
            severity = DrawdownSeverity.CRITICAL
        elif depth_exceeded or duration_exceeded:
            severity = DrawdownSeverity.WARNING
        else:
            severity = DrawdownSeverity.NORMAL

        return DrawdownCheckResult(
            severity=severity,
            current_depth=depth,
            current_duration_days=duration,
            expected_max_depth=exp_depth,
            expected_max_duration=exp_duration,
            depth_exceeded=depth_exceeded,
            duration_exceeded=duration_exceeded,
        )

    def expected_max_drawdown(self, n_days: int) -> float:
        """Magdon-Ismail 근사에 의한 기대 최대 drawdown.

        E[MDD] ≈ sigma * sqrt(n) * gamma(z) 형태 근사.

        For practical use, uses simplified bound:
            E[MDD] ≈ z * sigma * sqrt(n) (log-correction 포함)

        See: Magdon-Ismail et al. (2004), J. Appl. Probab. 41(1), 147-161.

        Args:
            n_days: 관측 일수

        Returns:
            기대 최대 drawdown (0.0 ~ 1.0)
        """
        if n_days <= 0:
            return 0.0

        sqrt_n = math.sqrt(n_days)
        vol_term = self._sigma * sqrt_n

        # Magdon-Ismail: E[MDD] = f(drift_ratio) * sigma * sqrt(n)
        # drift ratio alpha = mu / (0.5 * sigma^2)
        # For alpha < 0: MDD grows faster (趋势恶化)
        # For alpha > 0: MDD stabilizes
        drift_ratio = self._mu / (0.5 * self._sigma**2) if self._sigma > _EPSILON else 0.0

        if drift_ratio >= 0:
            # Positive drift: E[MDD] ≈ sigma * sqrt(2 * ln(n) / pi)
            log_term = math.log(max(n_days, 1))
            gamma_factor = math.sqrt(2.0 * log_term / math.pi)
            expected = vol_term * gamma_factor
        else:
            # Negative drift: linearly growing MDD approximation
            # E[MDD] ≈ |mu| * n + z * sigma * sqrt(n)
            expected = abs(self._mu) * n_days + self._z_score * vol_term

        # Confidence adjustment: scale by z-score / sqrt(2*ln(2))
        # This gives the confidence-level-adjusted expected max
        confidence_factor = self._z_score / math.sqrt(2.0 * math.log(2.0))
        expected *= confidence_factor

        return min(expected, 1.0)

    def expected_max_duration(self, n_days: int) -> int:
        """GBM 기반 기대 최대 drawdown 지속 기간.

        Rough heuristic based on mean-reverting drawdown duration:
            E[max_dur] ≈ n * sigma^2 / (2 * max(mu, sigma^2/2))

        For positive drift, drawdowns are shorter.
        For negative drift, duration can approach n.

        Args:
            n_days: 관측 일수

        Returns:
            기대 최대 drawdown 지속 일수
        """
        if n_days <= 0:
            return 0

        half_var = 0.5 * self._sigma**2
        effective_drift = max(self._mu, half_var)

        if effective_drift < _EPSILON:
            # Zero/negative drift: duration can be arbitrarily long
            return n_days

        # Base duration from drift/vol ratio
        base_duration = half_var / effective_drift * n_days

        # Confidence scaling: z * sqrt(n) correction
        correction = self._z_score * math.sqrt(n_days)

        result = base_duration + correction
        return min(int(result), n_days)

    def reset(self) -> None:
        """Monitor 상태를 초기화합니다."""
        self._peak_equity = 1.0
        self._current_equity = 1.0
        self._dd_start_day = 0
        self._n_days = 0
        self._in_drawdown = False

    def to_dict(self) -> dict[str, float | int | bool]:
        """Serialize mutable state for persistence."""
        return {
            "peak_equity": self._peak_equity,
            "current_equity": self._current_equity,
            "dd_start_day": self._dd_start_day,
            "n_days": self._n_days,
            "in_drawdown": self._in_drawdown,
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore mutable state from persisted dict."""
        self._peak_equity = float(data.get("peak_equity", 1.0))
        self._current_equity = float(data.get("current_equity", 1.0))
        self._dd_start_day = int(data.get("dd_start_day", 0))
        self._n_days = int(data.get("n_days", 0))
        self._in_drawdown = bool(data.get("in_drawdown", False))

    @staticmethod
    def estimate_params(daily_returns: list[float]) -> tuple[float, float]:
        """일별 수익률 리스트에서 GBM 파라미터 추정.

        Args:
            daily_returns: 일별 수익률 리스트

        Returns:
            (mu, sigma) 튜플. 데이터 부족 시 (0.0, 0.01)
        """
        valid = [r for r in daily_returns if math.isfinite(r)]
        if len(valid) < 2:  # noqa: PLR2004
            return 0.0, 0.01

        mu = sum(valid) / len(valid)
        var = sum((r - mu) ** 2 for r in valid) / (len(valid) - 1)
        sigma = math.sqrt(var)
        return mu, max(sigma, _EPSILON)


# ── Private Helpers ─────────────────────────────────────────────


def _norm_ppf(p: float) -> float:
    """Normal distribution percent point function (inverse CDF).

    Abramowitz & Stegun 26.2.23 rational approximation (|error| < 4.5e-4).
    scipy 무의존.

    Args:
        p: 확률 (0 < p < 1)

    Returns:
        z-score
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0

    # Symmetry: for p > 0.5, use 1-p and negate
    if p > 0.5:  # noqa: PLR2004
        return -_norm_ppf(1.0 - p)

    # Rational approximation for 0 < p <= 0.5
    t = math.sqrt(-2.0 * math.log(p))
    numerator = _PPF_A0 + _PPF_A1 * t + _PPF_A2 * t * t
    denominator = 1.0 + _PPF_B1 * t + _PPF_B2 * t * t + _PPF_B3 * t * t * t
    return -(t - numerator / denominator)
