"""Risk Aggregator — 포트폴리오 리스크 집계·검사.

포트폴리오 수준의 리스크 기여도, 분산도, 상관 스트레스를 계산하고,
OrchestratorConfig에 정의된 5가지 리스크 한도를 검사합니다.

Pure functions + thin RiskAggregator class (config 보유).

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - Zero-Tolerance Lint Policy
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.orchestrator.models import RiskAlert
from src.orchestrator.netting import compute_gross_leverage

if TYPE_CHECKING:
    import pandas as pd

    from src.orchestrator.config import OrchestratorConfig
    from src.orchestrator.models import PodPerformance

# ── Constants ─────────────────────────────────────────────────────

_MIN_VARIANCE = 1e-12
_WARNING_RATIO = 0.8  # 임계값의 80%에서 warning 발행
_MIN_PODS_FOR_CORRELATION = 2
_MIN_COV_ROWS = 3  # cov()/corr() 계산 최소 행 수
_MIN_VARIANCE_PER_POD = 1e-10  # Pod별 분산 임계값 (immature 판별)
_MIN_MATURE_PODS = 2  # PRC 의미 있는 계산 최소 mature pod 수


# ── Pure Functions ────────────────────────────────────────────────


def compute_risk_contributions(
    pod_returns: pd.DataFrame,
    weights: dict[str, float],
) -> dict[str, float]:
    """Pod별 Percentage Risk Contribution (PRC)을 계산합니다.

    PRC_i = w_i * (Σw)_i / (w^T Σ w)

    Args:
        pod_returns: columns=pod_ids, index=dates 일별 수익률
        weights: {pod_id: weight} 매핑

    Returns:
        {pod_id: PRC} 매핑. 합산 ≈ 1.0 (정상 시).
        weights가 비거나 분산이 0이면 균등 배분 반환.
    """
    pod_ids = [pid for pid in weights if pid in pod_returns.columns]
    if not pod_ids:
        return {}

    n = len(pod_ids)
    if len(pod_returns) < _MIN_COV_ROWS:
        equal = 1.0 / n if n > 0 else 0.0
        return dict.fromkeys(pod_ids, equal)

    # Immature pod 필터링: 분산이 극소인 Pod가 대다수면 equal fallback
    variances: pd.Series = pod_returns[pod_ids].var()  # type: ignore[assignment]
    mature_count = sum(1 for pid in pod_ids if float(variances[pid]) > _MIN_VARIANCE_PER_POD)
    if mature_count < _MIN_MATURE_PODS:
        equal = 1.0 / n if n > 0 else 0.0
        return dict.fromkeys(pod_ids, equal)

    w = np.array([weights[pid] for pid in pod_ids])
    cov: np.ndarray = pod_returns[pod_ids].cov().to_numpy()  # type: ignore[assignment]

    # NaN 처리
    cov = np.where(np.isfinite(cov), cov, 0.0)

    portfolio_var = float(w @ cov @ w)
    if portfolio_var < _MIN_VARIANCE:
        equal = 1.0 / n if n > 0 else 0.0
        return dict.fromkeys(pod_ids, equal)

    marginal = cov @ w  # Σw
    risk_contrib = w * marginal  # w_i * (Σw)_i
    prc = risk_contrib / portfolio_var

    return {pod_ids[i]: float(prc[i]) for i in range(n)}


def compute_effective_n(prc: dict[str, float]) -> float:
    """유효 분산 수 (Effective N = 1/HHI)를 계산합니다.

    Args:
        prc: {pod_id: PRC} 매핑

    Returns:
        유효 분산 수. PRC가 비거나 HHI가 0이면 0.0.
    """
    if not prc:
        return 0.0

    values = [abs(v) for v in prc.values()]
    total = sum(values)
    if total < _MIN_VARIANCE:
        return 0.0

    # Normalize
    normalized = [v / total for v in values]
    hhi = sum(v**2 for v in normalized)
    if hhi < _MIN_VARIANCE:
        return 0.0

    return 1.0 / hhi


def check_correlation_stress(
    pod_returns: pd.DataFrame,
    threshold: float,
) -> tuple[bool, float]:
    """Pod 간 상관 스트레스를 검사합니다.

    평균 pair-wise 상관계수가 threshold를 초과하면 스트레스 상태.

    Args:
        pod_returns: columns=pod_ids 일별 수익률
        threshold: 상관 스트레스 임계값 (예: 0.70)

    Returns:
        (is_stressed, avg_correlation) 튜플.
        Pod < 2개이면 (False, 0.0).
    """
    if pod_returns.shape[1] < _MIN_PODS_FOR_CORRELATION:
        return (False, 0.0)
    if pod_returns.shape[0] < _MIN_COV_ROWS:
        return (False, 0.0)

    corr_matrix = pod_returns.corr()
    n = corr_matrix.shape[0]

    # 상삼각 off-diagonal 평균
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    off_diag = corr_matrix.to_numpy()[mask]
    off_diag = off_diag[np.isfinite(off_diag)]

    if len(off_diag) == 0:
        return (False, 0.0)

    avg_corr = float(np.mean(off_diag))
    return (avg_corr >= threshold, avg_corr)


def check_asset_correlation_stress(
    price_history: dict[str, list[float]],
    threshold: float,
) -> tuple[bool, float]:
    """에셋 레벨 가격 수익률 상관 스트레스를 검사합니다.

    Pod 수 < 3일 때 에셋 수준으로 상관관계를 보완합니다.

    Args:
        price_history: {symbol: [close_prices]} 매핑
        threshold: 상관 스트레스 임계값

    Returns:
        (is_stressed, avg_correlation) 튜플.
        에셋 < 2개 또는 데이터 < 3행이면 (False, 0.0).
    """
    # 에셋 필터: 최소 2개 + 데이터 최소 3행
    symbols = [s for s, prices in price_history.items() if len(prices) >= _MIN_COV_ROWS]
    if len(symbols) < _MIN_PODS_FOR_CORRELATION:
        return (False, 0.0)

    # 가격 → 수익률 변환
    min_len = min(len(price_history[s]) for s in symbols)
    if min_len < _MIN_COV_ROWS:
        return (False, 0.0)

    returns_data: dict[str, list[float]] = {}
    for symbol in symbols:
        prices = price_history[symbol][-min_len:]
        returns_data[symbol] = [
            (prices[i] - prices[i - 1]) / prices[i - 1] if prices[i - 1] != 0 else 0.0
            for i in range(1, len(prices))
        ]

    if not returns_data or len(next(iter(returns_data.values()))) < _MIN_COV_ROWS - 1:
        return (False, 0.0)

    # 상관행렬 계산
    n = len(symbols)
    returns_matrix = np.array([returns_data[s] for s in symbols])
    corr_matrix = np.corrcoef(returns_matrix)

    if not np.all(np.isfinite(corr_matrix)):
        return (False, 0.0)

    # 상삼각 off-diagonal 평균
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    off_diag = corr_matrix[mask]
    off_diag = off_diag[np.isfinite(off_diag)]

    if len(off_diag) == 0:
        return (False, 0.0)

    avg_corr = float(np.mean(off_diag))
    return (avg_corr >= threshold, avg_corr)


def compute_portfolio_drawdown(
    pod_performances: dict[str, PodPerformance],
    weights: dict[str, float],
) -> float:
    """가중 평균 포트폴리오 현재 낙폭을 계산합니다.

    Args:
        pod_performances: {pod_id: PodPerformance}
        weights: {pod_id: weight}

    Returns:
        가중 평균 현재 낙폭 (양수, 예: 0.10 = -10%).
        weights 합이 0이면 0.0.
    """
    total_weight = sum(abs(weights.get(pid, 0.0)) for pid in pod_performances)
    if total_weight < _MIN_VARIANCE:
        return 0.0

    weighted_dd = sum(
        perf.current_drawdown * abs(weights.get(pid, 0.0)) for pid, perf in pod_performances.items()
    )
    return weighted_dd / total_weight


# ── RiskAggregator ────────────────────────────────────────────────


class RiskAggregator:
    """포트폴리오 리스크 한도 검사기.

    OrchestratorConfig에 정의된 5가지 리스크 한도를 검사하고
    RiskAlert 리스트를 반환합니다.

    Warning: 임계값의 80% 도달
    Critical: 임계값 100% 도달(초과)

    Args:
        config: OrchestratorConfig
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self._config = config

    def check_portfolio_limits(
        self,
        net_weights: dict[str, float],
        pod_performances: dict[str, PodPerformance],
        pod_weights: dict[str, float],
        pod_returns: pd.DataFrame | None = None,
        daily_pnl_pct: float = 0.0,
        asset_price_history: dict[str, list[float]] | None = None,
    ) -> list[RiskAlert]:
        """5가지 포트폴리오 리스크 한도를 검사합니다.

        Args:
            net_weights: {symbol: net_weight} 넷팅된 가중치
            pod_performances: {pod_id: PodPerformance}
            pod_weights: {pod_id: capital_fraction}
            pod_returns: Pod별 일별 수익률 DataFrame (PRC/상관 계산용)
            daily_pnl_pct: 오늘 실현+미실현 PnL 비율
            asset_price_history: {symbol: [close_prices]} — Pod < 3일 때 에셋 상관 보완

        Returns:
            RiskAlert 리스트 (경고 없으면 빈 리스트)
        """
        alerts: list[RiskAlert] = []

        self._check_gross_leverage(net_weights, alerts)
        self._check_portfolio_drawdown(pod_performances, pod_weights, alerts)
        self._check_daily_loss(daily_pnl_pct, alerts)

        if pod_returns is not None and not pod_returns.empty:
            self._check_single_pod_risk(pod_returns, pod_weights, alerts)
            self._check_correlation_stress(pod_returns, alerts)

            # Pod < 3 AND asset_price_history → 에셋 레벨 상관 보완
            n_pods = pod_returns.shape[1]
            if n_pods < _MIN_COV_ROWS and asset_price_history:
                self._check_asset_correlation_stress(asset_price_history, alerts)

        return alerts

    # ── Private checks ────────────────────────────────────────────

    def _check_gross_leverage(
        self,
        net_weights: dict[str, float],
        alerts: list[RiskAlert],
    ) -> None:
        gross = compute_gross_leverage(net_weights)
        threshold = self._config.max_gross_leverage
        self._emit_alert(
            alert_type="gross_leverage",
            current=gross,
            threshold=threshold,
            message_template="Gross leverage {current:.2f}x vs limit {threshold:.1f}x",
            alerts=alerts,
        )

    def _check_portfolio_drawdown(
        self,
        pod_performances: dict[str, PodPerformance],
        pod_weights: dict[str, float],
        alerts: list[RiskAlert],
    ) -> None:
        dd = compute_portfolio_drawdown(pod_performances, pod_weights)
        threshold = self._config.max_portfolio_drawdown
        self._emit_alert(
            alert_type="portfolio_drawdown",
            current=dd,
            threshold=threshold,
            message_template="Portfolio drawdown {current:.2%} vs limit {threshold:.1%}",
            alerts=alerts,
        )

    def _check_daily_loss(
        self,
        daily_pnl_pct: float,
        alerts: list[RiskAlert],
    ) -> None:
        # daily_pnl_pct < 0이면 손실 → abs로 비교
        loss = abs(min(daily_pnl_pct, 0.0))
        threshold = self._config.daily_loss_limit
        self._emit_alert(
            alert_type="daily_loss",
            current=loss,
            threshold=threshold,
            message_template="Daily loss {current:.2%} vs limit {threshold:.1%}",
            alerts=alerts,
        )

    def _check_single_pod_risk(
        self,
        pod_returns: pd.DataFrame,
        pod_weights: dict[str, float],
        alerts: list[RiskAlert],
    ) -> None:
        # Mature pod 부족 시 PRC 체크 무의미 → skip
        pod_ids = [pid for pid in pod_weights if pid in pod_returns.columns]
        if pod_ids:
            variances: pd.Series = pod_returns[pod_ids].var()  # type: ignore[assignment]
            mature = sum(1 for pid in pod_ids if float(variances[pid]) > _MIN_VARIANCE_PER_POD)
            if mature < _MIN_MATURE_PODS:
                return

        prc = compute_risk_contributions(pod_returns, pod_weights)
        threshold = self._config.max_single_pod_risk_pct

        for pod_id, contribution in prc.items():
            abs_contrib = abs(contribution)
            if abs_contrib >= threshold:
                alerts.append(
                    RiskAlert(
                        alert_type="single_pod_risk",
                        severity="critical",
                        message=f"Pod {pod_id} PRC {abs_contrib:.2%} vs limit {threshold:.1%}",
                        current_value=abs_contrib,
                        threshold=threshold,
                        pod_id=pod_id,
                    )
                )
            elif abs_contrib >= threshold * _WARNING_RATIO:
                alerts.append(
                    RiskAlert(
                        alert_type="single_pod_risk",
                        severity="warning",
                        message=f"Pod {pod_id} PRC {abs_contrib:.2%} approaching limit {threshold:.1%}",
                        current_value=abs_contrib,
                        threshold=threshold,
                        pod_id=pod_id,
                    )
                )

    def _check_correlation_stress(
        self,
        pod_returns: pd.DataFrame,
        alerts: list[RiskAlert],
    ) -> None:
        threshold = self._config.correlation_stress_threshold
        is_stressed, avg_corr = check_correlation_stress(pod_returns, threshold)

        if is_stressed:
            alerts.append(
                RiskAlert(
                    alert_type="correlation_stress",
                    severity="critical",
                    message=f"Avg correlation {avg_corr:.2%} vs threshold {threshold:.1%}",
                    current_value=avg_corr,
                    threshold=threshold,
                )
            )
        elif avg_corr >= threshold * _WARNING_RATIO:
            alerts.append(
                RiskAlert(
                    alert_type="correlation_stress",
                    severity="warning",
                    message=f"Avg correlation {avg_corr:.2%} approaching threshold {threshold:.1%}",
                    current_value=avg_corr,
                    threshold=threshold,
                )
            )

    def _check_asset_correlation_stress(
        self,
        asset_price_history: dict[str, list[float]],
        alerts: list[RiskAlert],
    ) -> None:
        threshold = self._config.correlation_stress_threshold
        is_stressed, avg_corr = check_asset_correlation_stress(asset_price_history, threshold)

        if is_stressed:
            alerts.append(
                RiskAlert(
                    alert_type="asset_correlation_stress",
                    severity="critical",
                    message=f"Asset avg correlation {avg_corr:.2%} vs threshold {threshold:.1%}",
                    current_value=avg_corr,
                    threshold=threshold,
                )
            )
        elif avg_corr >= threshold * _WARNING_RATIO:
            alerts.append(
                RiskAlert(
                    alert_type="asset_correlation_stress",
                    severity="warning",
                    message=f"Asset avg correlation {avg_corr:.2%} approaching threshold {threshold:.1%}",
                    current_value=avg_corr,
                    threshold=threshold,
                )
            )

    # ── Alert helper ──────────────────────────────────────────────

    def _emit_alert(
        self,
        alert_type: str,
        current: float,
        threshold: float,
        message_template: str,
        alerts: list[RiskAlert],
    ) -> None:
        """임계값 대비 current를 검사하여 warning/critical alert 발행."""
        if current >= threshold:
            alerts.append(
                RiskAlert(
                    alert_type=alert_type,
                    severity="critical",
                    message=message_template.format(current=current, threshold=threshold),
                    current_value=current,
                    threshold=threshold,
                )
            )
        elif current >= threshold * _WARNING_RATIO:
            alerts.append(
                RiskAlert(
                    alert_type=alert_type,
                    severity="warning",
                    message=message_template.format(current=current, threshold=threshold),
                    current_value=current,
                    threshold=threshold,
                )
            )
