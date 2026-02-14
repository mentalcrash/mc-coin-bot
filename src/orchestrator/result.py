"""OrchestratedResult — 멀티 전략 오케스트레이터 백테스트 결과.

포트폴리오/Pod별 메트릭, equity curve, 배분 이력, lifecycle 이벤트,
리스크 기여도를 담는 결과 컨테이너입니다.

Rules Applied:
    - #10 Python Standards: dataclass, type hints
    - #11 Pydantic: PerformanceMetrics 재사용
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.models.backtest import PerformanceMetrics


@dataclass
class OrchestratedResult:
    """멀티 전략 오케스트레이터 백테스트 결과.

    Attributes:
        portfolio_metrics: 전체 포트폴리오 성과 메트릭
        portfolio_equity_curve: 전체 포트폴리오 equity curve (timestamp index)
        pod_metrics: Pod별 간이 성과 메트릭 (pod_id → metrics dict)
        pod_equity_curves: Pod별 equity curve (pod_id → Series)
        allocation_history: 자본 배분 이력 (columns: timestamp + pod_ids)
        lifecycle_events: 생애주기 상태 전이 이력
        risk_contributions: 리스크 기여도 이력 (columns: timestamp + pod_ids)
    """

    portfolio_metrics: PerformanceMetrics
    portfolio_equity_curve: pd.Series
    pod_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    pod_equity_curves: dict[str, pd.Series] = field(default_factory=dict)
    allocation_history: pd.DataFrame | None = None
    lifecycle_events: list[dict[str, object]] = field(default_factory=list)
    risk_contributions: pd.DataFrame | None = None
