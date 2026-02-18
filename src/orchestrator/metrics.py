"""OrchestratorMetrics — Pod/Portfolio Prometheus 메트릭.

StrategyOrchestrator의 내부 상태(Pod 성과, 자본 배분, 생애주기, 리스크)를
Prometheus 메트릭으로 노출합니다.

메트릭 레이어:
    - Pod 레벨: equity, allocation, sharpe, drawdown, lifecycle, PRC
    - Portfolio 레벨: effective_n, avg_correlation, active_pods

Rules Applied:
    - Prometheus naming: mcbot_ prefix
    - #10 Python Standards: Modern typing, type hints
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger
from prometheus_client import Enum as PromEnum, Gauge

from src.orchestrator.netting import compute_netting_stats
from src.orchestrator.risk_aggregator import (
    check_correlation_stress,
    compute_effective_n,
    compute_risk_contributions,
)

if TYPE_CHECKING:
    from src.orchestrator.orchestrator import StrategyOrchestrator

# ── Pod-level Gauges ─────────────────────────────────────────────

pod_equity_gauge = Gauge(
    "mcbot_pod_equity_usdt",
    "Pod current equity (USDT)",
    ["pod_id"],
)
pod_allocation_gauge = Gauge(
    "mcbot_pod_allocation_fraction",
    "Pod capital allocation fraction",
    ["pod_id"],
)
pod_sharpe_gauge = Gauge(
    "mcbot_pod_rolling_sharpe",
    "Pod rolling Sharpe ratio",
    ["pod_id"],
)
pod_drawdown_gauge = Gauge(
    "mcbot_pod_drawdown_pct",
    "Pod current drawdown percentage",
    ["pod_id"],
)
pod_risk_contribution_gauge = Gauge(
    "mcbot_pod_risk_contribution",
    "Pod percentage risk contribution (PRC)",
    ["pod_id"],
)

# Lifecycle Enum (5 states)
_LIFECYCLE_STATES = ["incubation", "production", "warning", "probation", "retired"]

pod_lifecycle_enum = PromEnum(
    "mcbot_pod_lifecycle_state",
    "Pod lifecycle state",
    ["pod_id"],
    states=_LIFECYCLE_STATES,
)

# ── Portfolio-level Gauges ───────────────────────────────────────

portfolio_effective_n_gauge = Gauge(
    "mcbot_portfolio_effective_n",
    "Portfolio effective diversification (1/HHI)",
)
portfolio_avg_correlation_gauge = Gauge(
    "mcbot_portfolio_avg_correlation",
    "Portfolio average pair-wise correlation",
)
active_pods_gauge = Gauge(
    "mcbot_active_pods",
    "Number of active pods",
)

# ── Netting Gauges ────────────────────────────────────────────────

netting_gross_gauge = Gauge(
    "mcbot_netting_gross_exposure",
    "Total gross exposure before netting",
)
netting_net_gauge = Gauge(
    "mcbot_netting_net_exposure",
    "Total net exposure after netting",
)
netting_offset_ratio_gauge = Gauge(
    "mcbot_netting_offset_ratio",
    "Netting offset ratio (0=no offset, 1=full offset)",
)

# ── Constants ────────────────────────────────────────────────────

_MIN_PODS_FOR_PORTFOLIO = 2
_DEFAULT_CORRELATION_THRESHOLD = 1.0  # 임의 높은 값 (stress 판별 아닌 avg_corr 조회용)


# ── OrchestratorMetrics ─────────────────────────────────────────


class OrchestratorMetrics:
    """Orchestrator 상태를 Prometheus 메트릭으로 동기화.

    LiveRunner의 _periodic_metrics_update()에서 30초마다 update() 호출.

    Args:
        orchestrator: StrategyOrchestrator 인스턴스
    """

    def __init__(self, orchestrator: StrategyOrchestrator) -> None:
        self._orchestrator = orchestrator

    def update(self) -> None:
        """모든 Pod + Portfolio + Netting + Anomaly 메트릭 갱신."""
        try:
            self._update_pod_metrics()
            self._update_portfolio_metrics()
            self._update_netting_metrics()
            self._update_anomaly_metrics()
        except Exception:
            logger.exception("OrchestratorMetrics update failed")

    def _update_pod_metrics(self) -> None:
        """Pod별 Gauge/Enum 업데이트."""
        for pod in self._orchestrator.pods:
            pid = pod.pod_id
            pod_equity_gauge.labels(pod_id=pid).set(pod.performance.current_equity)
            pod_allocation_gauge.labels(pod_id=pid).set(pod.capital_fraction)
            pod_sharpe_gauge.labels(pod_id=pid).set(pod.performance.sharpe_ratio)
            pod_drawdown_gauge.labels(pod_id=pid).set(pod.performance.current_drawdown)
            pod_lifecycle_enum.labels(pod_id=pid).state(pod.state.value)

        active_pods_gauge.set(self._orchestrator.active_pod_count)

    def _update_portfolio_metrics(self) -> None:
        """Portfolio-level 메트릭 업데이트 (PRC, Effective N, Avg Correlation)."""
        active_pods = [p for p in self._orchestrator.pods if p.is_active]

        if len(active_pods) < _MIN_PODS_FOR_PORTFOLIO:
            # Pod 부족 시 PRC = 균등, portfolio 메트릭 0
            for pod in active_pods:
                equal_prc = 1.0 / len(active_pods) if active_pods else 0.0
                pod_risk_contribution_gauge.labels(pod_id=pod.pod_id).set(equal_prc)
            portfolio_effective_n_gauge.set(float(len(active_pods)))
            portfolio_avg_correlation_gauge.set(0.0)
            return

        # Pod 수익률 DataFrame 구성
        pod_returns_data: dict[str, list[float]] = {}
        weights: dict[str, float] = {}
        for pod in active_pods:
            returns = list(pod.daily_returns_series)
            pod_returns_data[pod.pod_id] = returns if returns else [0.0]
            weights[pod.pod_id] = pod.capital_fraction

        max_len = max(len(v) for v in pod_returns_data.values())
        for pid, current in pod_returns_data.items():
            if len(current) < max_len:
                pod_returns_data[pid] = [0.0] * (max_len - len(current)) + current

        pod_returns = pd.DataFrame(pod_returns_data)

        # PRC 계산
        prc = compute_risk_contributions(pod_returns, weights)
        for pid, contribution in prc.items():
            pod_risk_contribution_gauge.labels(pod_id=pid).set(contribution)

        # Effective N
        eff_n = compute_effective_n(prc)
        portfolio_effective_n_gauge.set(eff_n)

        # Avg Correlation
        _, avg_corr = check_correlation_stress(pod_returns, _DEFAULT_CORRELATION_THRESHOLD)
        portfolio_avg_correlation_gauge.set(avg_corr)

    def _update_anomaly_metrics(self) -> None:
        """Anomaly detection 결과를 Prometheus gauge로 export."""
        lifecycle = self._orchestrator.lifecycle
        if lifecycle is None:
            return

        from src.monitoring.metrics import (
            distribution_ks_statistic_gauge,
            distribution_p_value_gauge,
            gbm_drawdown_depth_gauge,
            gbm_drawdown_duration_gauge,
            gbm_severity_gauge,
            ransac_conformal_lower_gauge,
            ransac_current_cumulative_gauge,
            ransac_decay_detected_gauge,
            ransac_slope_gauge,
        )

        gbm_severity_map = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}

        for pod in self._orchestrator.pods:
            pid = pod.pod_id
            if not pod.is_active:
                continue

            # Distribution drift
            dist_result = lifecycle.get_distribution_result(pid)
            if dist_result is not None:
                distribution_ks_statistic_gauge.labels(strategy=pid).set(dist_result.ks_statistic)
                distribution_p_value_gauge.labels(strategy=pid).set(dist_result.p_value)

            # RANSAC decay
            ransac_result = lifecycle.get_ransac_result(pid)
            if ransac_result is not None:
                ransac_slope_gauge.labels(strategy=pid).set(ransac_result.ransac_slope)
                ransac_conformal_lower_gauge.labels(strategy=pid).set(
                    ransac_result.conformal_lower_bound
                )
                ransac_decay_detected_gauge.labels(strategy=pid).set(
                    1.0 if (ransac_result.level_breach or not ransac_result.slope_positive) else 0.0
                )
                ransac_current_cumulative_gauge.labels(strategy=pid).set(
                    ransac_result.current_cumulative
                )

            # GBM drawdown
            gbm_result = lifecycle.get_gbm_result(pid)
            if gbm_result is not None:
                gbm_drawdown_depth_gauge.labels(strategy=pid).set(gbm_result.current_depth)
                gbm_drawdown_duration_gauge.labels(strategy=pid).set(
                    gbm_result.current_duration_days
                )
                gbm_severity_gauge.labels(strategy=pid).set(
                    gbm_severity_map.get(gbm_result.severity.value, 0)
                )

    def _update_netting_metrics(self) -> None:
        """Netting 상쇄 메트릭 업데이트."""
        pod_targets = self._orchestrator.last_pod_targets
        if not pod_targets:
            netting_gross_gauge.set(0.0)
            netting_net_gauge.set(0.0)
            netting_offset_ratio_gauge.set(0.0)
            return

        stats = compute_netting_stats(pod_targets)
        netting_gross_gauge.set(stats.gross_sum)
        netting_net_gauge.set(stats.net_sum)
        netting_offset_ratio_gauge.set(stats.offset_ratio)
