"""Health Check 알림 데이터 모델.

HealthCheckScheduler에서 수집한 시스템/마켓/전략 상태를 표현하는
Pydantic frozen 모델입니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, ConfigDict
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SystemHealthSnapshot(BaseModel):
    """Tier 1: System Heartbeat 데이터.

    Attributes:
        timestamp: 수집 시각
        uptime_seconds: 봇 가동 시간 (초)
        total_equity: 총 자산
        available_cash: 가용 현금
        aggregate_leverage: 합산 레버리지
        open_position_count: 오픈 포지션 수
        total_symbols: 전체 심볼 수
        current_drawdown: 현재 drawdown (0.0~1.0)
        peak_equity: 고점 equity
        is_circuit_breaker_active: CB 발동 여부
        today_pnl: 금일 실현 PnL
        today_trades: 금일 거래 건수
        stale_symbol_count: stale 심볼 수
        bars_emitted: 발행된 bar 이벤트 수
        events_dropped: 드롭된 이벤트 수
        max_queue_depth: 최대 큐 깊이
        is_notification_degraded: 알림 시스템 degraded 여부
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    uptime_seconds: float
    total_equity: float
    available_cash: float
    aggregate_leverage: float
    open_position_count: int
    total_symbols: int
    current_drawdown: float
    peak_equity: float
    is_circuit_breaker_active: bool
    today_pnl: float
    today_trades: int
    stale_symbol_count: int
    bars_emitted: int
    events_dropped: int
    max_queue_depth: int
    is_notification_degraded: bool


class SymbolDerivativesSnapshot(BaseModel):
    """개별 심볼의 파생상품 데이터 스냅샷.

    Attributes:
        symbol: 거래 심볼
        price: 현재 가격
        funding_rate: 최근 funding rate
        funding_rate_annualized: 연환산 funding rate (%)
        open_interest: 미결제약정 (USDT)
        ls_ratio: Long/Short 계정 비율
        taker_ratio: Taker Buy/Sell 비율
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    price: float
    funding_rate: float
    funding_rate_annualized: float
    open_interest: float
    ls_ratio: float
    taker_ratio: float


class MarketRegimeReport(BaseModel):
    """Tier 2: Market Regime 리포트.

    Attributes:
        timestamp: 수집 시각
        regime_score: 종합 regime 점수 (-1.0 ~ +1.0)
        regime_label: 해석 라벨 (Extreme Greed, Bullish, ...)
        symbols: 심볼별 스냅샷 목록
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    regime_score: float
    regime_label: str
    symbols: tuple[SymbolDerivativesSnapshot, ...]


class PositionStatus(BaseModel):
    """오픈 포지션 요약.

    Attributes:
        symbol: 심볼
        direction: 방향 (LONG/SHORT)
        unrealized_pnl: 미실현 PnL
        size: 포지션 수량
        current_weight: 포트폴리오 비중
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    direction: str
    unrealized_pnl: float
    size: float
    current_weight: float


class StrategyHealthSnapshot(BaseModel):
    """Tier 3: Strategy Health 리포트.

    Attributes:
        timestamp: 수집 시각
        rolling_sharpe_30d: 최근 30일 rolling Sharpe
        win_rate_recent: 최근 20건 win rate
        profit_factor: 최근 20건 profit factor
        total_closed_trades: 전체 종결 거래 수
        open_positions: 오픈 포지션 목록
        is_circuit_breaker_active: CB 발동 여부
        alpha_decay_detected: 3연속 Sharpe 하락 감지
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    rolling_sharpe_30d: float
    win_rate_recent: float
    profit_factor: float
    total_closed_trades: int
    open_positions: tuple[PositionStatus, ...]
    is_circuit_breaker_active: bool
    alpha_decay_detected: bool
