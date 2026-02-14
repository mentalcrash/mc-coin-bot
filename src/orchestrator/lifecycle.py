"""Lifecycle Manager — Pod 생애주기 자동 관리.

Pod의 성과를 실시간으로 평가하여 상태 전이를 수행합니다.

State Machine:
    INCUBATION → PRODUCTION  (졸업 기준 ALL 충족)
    PRODUCTION → WARNING     (PH detector 열화 감지)
    WARNING    → PRODUCTION  (PH score 회복)
    WARNING    → PROBATION   (30일 미회복)
    PROBATION  → PRODUCTION  (강한 회복)
    PROBATION  → RETIRED     (유예 만료)
    ANY        → RETIRED     (hard stops: MDD ≥ 25% 또는 연속 6개월 손실)

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - PLR0912: 서브메서드 분리로 branch count 제한
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.orchestrator.degradation import PageHinkleyDetector
from src.orchestrator.models import LifecycleState

if TYPE_CHECKING:
    import pandas as pd

    from src.orchestrator.config import GraduationCriteria, RetirementCriteria
    from src.orchestrator.pod import StrategyPod


# ── Constants ─────────────────────────────────────────────────────

_DAYS_PER_MONTH = 30
_WARNING_TIMEOUT_DAYS = 30
_MIN_CORRELATION_SAMPLES = 2
_MIN_WARNING_OBSERVATION_DAYS = 5
_PH_RECOVERY_RATIO = 0.2


# ── Internal State ────────────────────────────────────────────────


@dataclass
class _PodLifecycleState:
    """Pod별 내부 생애주기 상태 추적."""

    ph_detector: PageHinkleyDetector = field(default_factory=PageHinkleyDetector)
    state_entered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    consecutive_loss_months: int = 0
    last_monthly_check_day: int = 0


# ── LifecycleManager ─────────────────────────────────────────────


class LifecycleManager:
    """Pod 생애주기 자동 관리자.

    각 Pod의 성과를 평가하여 상태 전이를 수행합니다.
    Orchestrator._execute_rebalance()에서 호출됩니다.

    Args:
        graduation: Incubation → Production 졸업 기준
        retirement: 퇴출 기준 (hard stops + probation)
    """

    def __init__(
        self,
        graduation: GraduationCriteria,
        retirement: RetirementCriteria,
    ) -> None:
        self._graduation = graduation
        self._retirement = retirement
        self._pod_states: dict[str, _PodLifecycleState] = {}

    def evaluate(
        self,
        pod: StrategyPod,
        portfolio_returns: pd.Series | None = None,
        *,
        bar_timestamp: datetime | None = None,
    ) -> LifecycleState:
        """Pod 상태를 평가하고 필요 시 전이를 수행합니다.

        Args:
            pod: 평가 대상 StrategyPod
            portfolio_returns: 전체 포트폴리오 일별 수익률 (correlation 계산용)
            bar_timestamp: 현재 bar의 타임스탬프 (None이면 wall clock 사용)

        Returns:
            평가 후 Pod의 LifecycleState
        """
        pod_ls = self._get_or_create_state(pod)
        _now = bar_timestamp or datetime.now(UTC)

        # 1. RETIRED는 terminal → 즉시 반환
        if pod.state == LifecycleState.RETIRED:
            return LifecycleState.RETIRED

        # 2. 연속 손실 개월 업데이트
        self._update_consecutive_loss_months(pod, pod_ls)

        # 3. Hard stops 체크 (ANY → RETIRED)
        if self._check_hard_stops(pod, pod_ls, now=_now):
            return pod.state

        # 4. 상태별 전이 평가
        self._evaluate_state_specific(pod, pod_ls, portfolio_returns, now=_now)

        return pod.state

    # ── State-specific dispatchers ────────────────────────────────

    def _evaluate_state_specific(
        self,
        pod: StrategyPod,
        pod_ls: _PodLifecycleState,
        portfolio_returns: pd.Series | None,
        *,
        now: datetime,
    ) -> None:
        """상태별 전이 로직 디스패치 (PLR0912 대응)."""
        state = pod.state
        if state == LifecycleState.INCUBATION:
            self._evaluate_incubation(pod, pod_ls, portfolio_returns, now=now)
        elif state == LifecycleState.PRODUCTION:
            self._evaluate_production(pod, pod_ls, now=now)
        elif state == LifecycleState.WARNING:
            self._evaluate_warning(pod, pod_ls, now=now)
        elif state == LifecycleState.PROBATION:
            self._evaluate_probation(pod, pod_ls, now=now)

    def _evaluate_incubation(
        self,
        pod: StrategyPod,
        pod_ls: _PodLifecycleState,
        portfolio_returns: pd.Series | None,
        *,
        now: datetime,
    ) -> None:
        """INCUBATION → PRODUCTION 졸업 평가."""
        if self._check_graduation(pod, portfolio_returns):
            self._transition(pod, pod_ls, LifecycleState.PRODUCTION, now=now)

    def _evaluate_production(
        self,
        pod: StrategyPod,
        pod_ls: _PodLifecycleState,
        *,
        now: datetime,
    ) -> None:
        """PRODUCTION → WARNING 열화 평가."""
        if self._check_degradation(pod, pod_ls):
            self._transition(pod, pod_ls, LifecycleState.WARNING, now=now)

    def _evaluate_warning(
        self,
        pod: StrategyPod,
        pod_ls: _PodLifecycleState,
        *,
        now: datetime,
    ) -> None:
        """WARNING → PRODUCTION (recovery) 또는 → PROBATION (timeout)."""
        days_in_warning = (now - pod_ls.state_entered_at).days

        # Recovery: PH score < lambda * 0.2 AND 최소 관찰 기간 경과
        ph_recovery_threshold = pod_ls.ph_detector.lambda_threshold * _PH_RECOVERY_RATIO
        if (
            days_in_warning >= _MIN_WARNING_OBSERVATION_DAYS
            and pod_ls.ph_detector.score < ph_recovery_threshold
        ):
            self._transition(pod, pod_ls, LifecycleState.PRODUCTION, now=now)
            return

        # Timeout: 30 days without recovery → PROBATION
        if days_in_warning >= _WARNING_TIMEOUT_DAYS:
            self._transition(pod, pod_ls, LifecycleState.PROBATION, now=now)

    def _evaluate_probation(
        self,
        pod: StrategyPod,
        pod_ls: _PodLifecycleState,
        *,
        now: datetime,
    ) -> None:
        """PROBATION → PRODUCTION (strong recovery) 또는 → RETIRED (expired)."""
        # Strong recovery: sharpe >= min_sharpe AND ph_score <= 0
        perf = pod.performance
        if perf.sharpe_ratio >= self._graduation.min_sharpe and pod_ls.ph_detector.score <= 0.0:
            self._transition(pod, pod_ls, LifecycleState.PRODUCTION, now=now)
            return

        # Expired: probation_days 경과 → RETIRED
        days_in_probation = (now - pod_ls.state_entered_at).days
        if days_in_probation >= self._retirement.probation_days:
            self._transition(pod, pod_ls, LifecycleState.RETIRED, now=now)

    # ── Hard Stops ────────────────────────────────────────────────

    def _check_hard_stops(
        self,
        pod: StrategyPod,
        pod_ls: _PodLifecycleState,
        *,
        now: datetime,
    ) -> bool:
        """MDD breach 또는 연속 손실 개월 초과 시 즉시 RETIRED.

        Returns:
            True if hard stop triggered (state changed to RETIRED).
        """
        perf = pod.performance

        # MDD breach
        if perf.max_drawdown >= self._retirement.max_drawdown_breach:
            logger.warning(
                "Pod {}: MDD {:.1%} >= {:.1%} breach → RETIRED",
                pod.pod_id,
                perf.max_drawdown,
                self._retirement.max_drawdown_breach,
            )
            self._transition(pod, pod_ls, LifecycleState.RETIRED, now=now)
            return True

        # Consecutive loss months
        if pod_ls.consecutive_loss_months >= self._retirement.consecutive_loss_months:
            logger.warning(
                "Pod {}: {} consecutive loss months → RETIRED",
                pod.pod_id,
                pod_ls.consecutive_loss_months,
            )
            self._transition(pod, pod_ls, LifecycleState.RETIRED, now=now)
            return True

        return False

    # ── Graduation Check ──────────────────────────────────────────

    def _check_graduation(
        self,
        pod: StrategyPod,
        portfolio_returns: pd.Series | None,
    ) -> bool:
        """7개 졸업 기준 중 6개 평가 (backtest_live_gap deferred).

        Returns:
            True if all criteria met.
        """
        perf = pod.performance
        grad = self._graduation

        # 5 quantitative criteria (backtest_live_gap deferred)
        quant_ok = (
            perf.live_days >= grad.min_live_days
            and perf.sharpe_ratio >= grad.min_sharpe
            and perf.max_drawdown <= grad.max_drawdown
            and perf.trade_count >= grad.min_trade_count
            and perf.calmar_ratio >= grad.min_calmar
        )
        if not quant_ok:
            return False

        # Portfolio correlation (skip if no portfolio returns)
        if portfolio_returns is not None:
            corr = self._compute_portfolio_correlation(pod, portfolio_returns)
            if corr > grad.max_portfolio_correlation:
                return False

        return True

    # ── Degradation Check ─────────────────────────────────────────

    def _check_degradation(
        self,
        pod: StrategyPod,
        pod_ls: _PodLifecycleState,
    ) -> bool:
        """PH detector에 최신 daily return 입력 → 열화 감지.

        Returns:
            True if degradation detected.
        """
        returns = pod.daily_returns
        if not returns:
            return False

        # Feed only the latest return
        latest_return = returns[-1]
        return pod_ls.ph_detector.update(latest_return)

    # ── Consecutive Loss Months ───────────────────────────────────

    def _update_consecutive_loss_months(
        self,
        pod: StrategyPod,
        pod_ls: _PodLifecycleState,
    ) -> None:
        """일별 수익률을 30일 청크로 분할하여 연속 손실 개월 카운트."""
        live_days = pod.performance.live_days

        # 새 30일 청크가 완료되지 않았으면 skip
        if live_days < _DAYS_PER_MONTH:
            return
        latest_month_idx = live_days // _DAYS_PER_MONTH
        if latest_month_idx <= pod_ls.last_monthly_check_day:
            return

        returns = pod.daily_returns

        # Process all pending monthly chunks
        start_idx = pod_ls.last_monthly_check_day + 1
        for month_idx in range(start_idx, latest_month_idx + 1):
            chunk_start = (month_idx - 1) * _DAYS_PER_MONTH
            chunk_end = month_idx * _DAYS_PER_MONTH
            chunk = returns[chunk_start:chunk_end]

            if not chunk:
                continue

            monthly_return = math.prod(1.0 + r for r in chunk) - 1.0
            if monthly_return < 0.0:
                pod_ls.consecutive_loss_months += 1
            else:
                pod_ls.consecutive_loss_months = 0

        pod_ls.last_monthly_check_day = latest_month_idx

    # ── Portfolio Correlation ─────────────────────────────────────

    def _compute_portfolio_correlation(
        self,
        pod: StrategyPod,
        portfolio_returns: pd.Series,
    ) -> float:
        """Pod returns vs portfolio returns 상관계수.

        Returns:
            Pearson correlation coefficient. 0.0 if insufficient data.
        """
        import pandas as pd

        pod_returns = pd.Series(pod.daily_returns, dtype=float)

        if (
            len(pod_returns) < _MIN_CORRELATION_SAMPLES
            or len(portfolio_returns) < _MIN_CORRELATION_SAMPLES
        ):
            return 0.0

        # Align lengths (use shorter)
        min_len = min(len(pod_returns), len(portfolio_returns))
        pod_tail = pod_returns.iloc[-min_len:]
        port_tail = portfolio_returns.iloc[-min_len:]

        # Reset index for alignment
        pod_tail = pod_tail.reset_index(drop=True)
        port_tail = port_tail.reset_index(drop=True)

        corr = pod_tail.corr(port_tail)

        # NaN → 0.0 (e.g. zero variance)
        if pd.isna(corr):
            return 0.0

        return float(corr)

    # ── State Transition ──────────────────────────────────────────

    def _transition(
        self,
        pod: StrategyPod,
        pod_ls: _PodLifecycleState,
        new_state: LifecycleState,
        *,
        now: datetime,
    ) -> None:
        """상태 전이를 수행합니다."""
        old_state = pod.state
        pod.state = new_state
        pod_ls.state_entered_at = now

        # WARNING/PROBATION → PRODUCTION 복귀 시 detector reset
        if (
            old_state in (LifecycleState.WARNING, LifecycleState.PROBATION)
            and new_state == LifecycleState.PRODUCTION
        ):
            pod_ls.ph_detector.reset()

        logger.info(
            "Pod {}: {} → {}",
            pod.pod_id,
            old_state.value,
            new_state.value,
        )

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Serialize lifecycle state for all tracked pods."""
        result: dict[str, dict[str, object]] = {}
        for pod_id, pls in self._pod_states.items():
            result[pod_id] = {
                "state_entered_at": pls.state_entered_at.isoformat(),
                "consecutive_loss_months": pls.consecutive_loss_months,
                "last_monthly_check_day": pls.last_monthly_check_day,
                "ph_detector": pls.ph_detector.to_dict(),
            }
        return result

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore lifecycle state from persisted dict.

        Pod IDs not in data are left at defaults. Unrecognized IDs are ignored.
        """
        for pod_id, pls_data in data.items():
            if not isinstance(pls_data, dict):
                continue

            pls = self._pod_states.get(pod_id)
            if pls is None:
                pls = _PodLifecycleState()
                self._pod_states[pod_id] = pls

            entered_at = pls_data.get("state_entered_at")
            if isinstance(entered_at, str):
                pls.state_entered_at = datetime.fromisoformat(entered_at)

            loss_months = pls_data.get("consecutive_loss_months")
            if isinstance(loss_months, int | float):
                pls.consecutive_loss_months = int(loss_months)

            check_day = pls_data.get("last_monthly_check_day")
            if isinstance(check_day, int | float):
                pls.last_monthly_check_day = int(check_day)

            ph_data = pls_data.get("ph_detector")
            if isinstance(ph_data, dict):
                pls.ph_detector.restore_from_dict(ph_data)

    # ── Internal ──────────────────────────────────────────────────

    def _get_or_create_state(self, pod: StrategyPod) -> _PodLifecycleState:
        """Pod별 내부 상태를 가져오거나 생성합니다."""
        if pod.pod_id not in self._pod_states:
            self._pod_states[pod.pod_id] = _PodLifecycleState()
        return self._pod_states[pod.pod_id]
