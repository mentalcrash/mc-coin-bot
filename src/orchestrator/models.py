"""Orchestrator Domain Models.

Strategy Orchestrator의 핵심 열거형과 런타임 데이터 컨테이너를 정의합니다.

Rules Applied:
    - #10 Python Standards: Modern typing (StrEnum), dataclass
    - #12 Data Engineering: UTC datetime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class LifecycleState(StrEnum):
    """Pod 생애주기 상태.

    Attributes:
        INCUBATION: 초기 관찰 기간 (소규모 자본)
        PRODUCTION: 본격 운용 중
        WARNING: 성과 악화 경고
        PROBATION: 유예 기간 (퇴출 임박)
        RETIRED: 운용 종료 / 퇴출
    """

    INCUBATION = "incubation"
    PRODUCTION = "production"
    WARNING = "warning"
    PROBATION = "probation"
    RETIRED = "retired"


class AllocationMethod(StrEnum):
    """자본 배분 방법.

    Attributes:
        EQUAL_WEIGHT: 동일 비중 배분
        RISK_PARITY: 리스크 기여도 균등 배분
        ADAPTIVE_KELLY: 적응적 Kelly 비율
        INVERSE_VOLATILITY: 변동성 역비례 배분
    """

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    ADAPTIVE_KELLY = "adaptive_kelly"
    INVERSE_VOLATILITY = "inverse_volatility"


class RebalanceTrigger(StrEnum):
    """리밸런싱 트리거 유형.

    Attributes:
        CALENDAR: 정기 리밸런싱 (N일 간격)
        THRESHOLD: 가중치 이탈 시 리밸런싱
        HYBRID: 정기 + 이탈 조합
    """

    CALENDAR = "calendar"
    THRESHOLD = "threshold"
    HYBRID = "hybrid"


@dataclass
class PodPerformance:
    """Pod별 성과 추적 컨테이너.

    매 bar마다 업데이트되는 mutable 런타임 객체입니다.

    Attributes:
        pod_id: Pod 식별자
        total_return: 누적 수익률
        sharpe_ratio: Sharpe 비율
        max_drawdown: 최대 낙폭 (양수 표현, 예: 0.15 = -15%)
        calmar_ratio: Calmar 비율
        win_rate: 승률
        trade_count: 총 거래 수
        live_days: 실운용 일수
        rolling_volatility: 롤링 변동성
        peak_equity: 고점 자본
        current_equity: 현재 자본
        current_drawdown: 현재 낙폭 (양수 표현)
        last_updated: 마지막 업데이트 시각 (UTC)
    """

    pod_id: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    live_days: int = 0
    rolling_volatility: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_profitable(self) -> bool:
        """누적 수익이 양수인지 여부."""
        return self.total_return > 0.0

    @property
    def equity_ratio(self) -> float:
        """현재 자본 / 고점 자본 비율.

        peak_equity가 0이면 0.0 반환 (division by zero 방지).
        """
        if self.peak_equity == 0.0:
            return 0.0
        return self.current_equity / self.peak_equity


@dataclass
class PodPosition:
    """Pod별 심볼 포지션 정보.

    Attributes:
        pod_id: Pod 식별자
        symbol: 거래 심볼 (예: "BTC/USDT")
        target_weight: Pod 내 목표 가중치
        global_weight: 전체 포트폴리오 기준 가중치
        notional_usd: 명목 가치 (USD)
        unrealized_pnl: 미실현 손익
        realized_pnl: 실현 손익
    """

    pod_id: str
    symbol: str
    target_weight: float = 0.0
    global_weight: float = 0.0
    notional_usd: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        """미실현 + 실현 손익 합계."""
        return self.unrealized_pnl + self.realized_pnl

    @property
    def is_open(self) -> bool:
        """포지션이 열려 있는지 여부 (notional > 0)."""
        return abs(self.notional_usd) > 0.0


@dataclass(frozen=True)
class RiskAlert:
    """포트폴리오 리스크 경고.

    RiskAggregator가 발행하는 immutable 경고 객체입니다.

    Attributes:
        alert_type: 경고 유형 (gross_leverage, portfolio_drawdown, daily_loss,
                    single_pod_risk, correlation_stress)
        severity: 심각도 ("warning" | "critical")
        message: 상세 메시지
        current_value: 현재 측정값
        threshold: 설정 임계값
        pod_id: 관련 Pod ID (None = 포트폴리오 전체)
    """

    alert_type: str
    severity: str
    message: str
    current_value: float
    threshold: float
    pod_id: str | None = None

    @staticmethod
    def has_critical(alerts: list[RiskAlert]) -> bool:
        """리스트에 critical 경고가 존재하는지 여부."""
        return any(a.severity == "critical" for a in alerts)
