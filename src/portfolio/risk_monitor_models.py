"""Portfolio Risk Monitor models.

포트폴리오 수준의 실시간 리스크 감시에 사용되는 모델 정의.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True
    - StrEnum for risk actions
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RiskAction(StrEnum):
    """포트폴리오 리스크 감시 결과 액션."""

    NORMAL = "normal"
    HALT_NEW_ENTRIES = "halt_new_entries"
    REDUCE_CORRELATED = "reduce_correlated"
    LIQUIDATE_ALL = "liquidate_all"


class PortfolioRiskConfig(BaseModel):
    """포트폴리오 리스크 모니터 설정.

    Attributes:
        max_portfolio_drawdown: -20% 초과 시 전체 청산
        max_daily_loss: -5% 초과 시 신규 진입 중단
        max_correlation_exposure: 동적 상관 임계값
        max_concentration_pct: 단일 심볼 집중 한도
    """

    model_config = ConfigDict(frozen=True)

    max_portfolio_drawdown: float = Field(default=0.20, description="HWM 대비 최대 허용 낙폭")
    max_daily_loss: float = Field(default=0.05, description="일일 최대 허용 손실")
    max_correlation_exposure: float = Field(default=0.70, description="평균 상관계수 임계값")
    max_concentration_pct: float = Field(default=0.40, description="단일 심볼 최대 비중")


class PortfolioRiskSnapshot(BaseModel):
    """포트폴리오 리스크 상태 스냅샷.

    Attributes:
        timestamp: 스냅샷 시각
        action: 판정된 액션
        portfolio_drawdown: 현재 포트폴리오 낙폭 (0~1)
        daily_pnl_pct: 당일 PnL 비율
        max_concentration: 최대 단일 심볼 비중 (0~1)
        avg_correlation: 평균 상관계수
        messages: 경고 메시지 목록
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    action: RiskAction
    portfolio_drawdown: float = 0.0
    daily_pnl_pct: float = 0.0
    max_concentration: float = 0.0
    avg_correlation: float = 0.0
    messages: list[str] = Field(default_factory=list)
