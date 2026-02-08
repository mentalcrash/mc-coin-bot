"""EDA 이벤트 타입 계층.

이 모듈은 Event-Driven Architecture에서 사용되는 모든 이벤트 타입을 정의합니다.
모든 이벤트는 불변(frozen)이며, correlation_id로 인과 관계를 추적합니다.

이벤트 체인:
    DataFeed → BarEvent → StrategyEngine → SignalEvent → PM → OrderRequestEvent
    → RM → OMS → FillEvent → PositionUpdateEvent → BalanceUpdateEvent

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Field validators
    - #10 Python Standards: Modern typing (Literal, StrEnum)
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.models.types import Direction


# ==========================================================================
# EventType Enum
# ==========================================================================
class EventType(StrEnum):
    """시스템 내 모든 이벤트 유형."""

    # Market Data
    BAR = "bar"

    # Strategy
    SIGNAL = "signal"

    # Execution
    ORDER_REQUEST = "order_request"
    ORDER_ACK = "order_ack"
    ORDER_REJECTED = "order_rejected"
    FILL = "fill"

    # Portfolio
    POSITION_UPDATE = "position_update"
    BALANCE_UPDATE = "balance_update"

    # Risk
    RISK_ALERT = "risk_alert"
    CIRCUIT_BREAKER = "circuit_breaker"

    # System
    HEARTBEAT = "heartbeat"


# ==========================================================================
# Market Data Events
# ==========================================================================
class BarEvent(BaseModel):
    """단일 OHLCV 캔들 이벤트.

    DataFeed가 발행하며, StrategyEngine과 AnalyticsEngine이 구독합니다.

    Attributes:
        symbol: 거래 심볼 (예: "BTC/USDT")
        timeframe: 타임프레임 (예: "1D", "4h")
        open: 시가
        high: 고가
        low: 저가
        close: 종가
        volume: 거래량
        bar_timestamp: 캔들 종료 시각
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.BAR
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_timestamp: datetime


# ==========================================================================
# Strategy Events
# ==========================================================================
class SignalEvent(BaseModel):
    """전략 시그널 이벤트.

    StrategyEngine이 발행하며, PortfolioManager가 구독합니다.

    Attributes:
        symbol: 거래 심볼
        strategy_name: 전략 이름
        direction: 매매 방향 (LONG/SHORT/NEUTRAL)
        strength: 시그널 강도 (unbounded, PM에서 클램핑)
        bar_timestamp: 시그널이 참조하는 캔들 시각
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.SIGNAL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    symbol: str
    strategy_name: str
    direction: Direction
    strength: float
    bar_timestamp: datetime


# ==========================================================================
# Execution Events
# ==========================================================================
class OrderRequestEvent(BaseModel):
    """주문 요청 이벤트.

    PM이 생성하고, RM이 검증한 후 OMS로 전달합니다.

    Attributes:
        client_order_id: 멱등성 키
        symbol: 거래 심볼
        side: 매수/매도
        order_type: 주문 유형 (MARKET/LIMIT)
        target_weight: 목표 포트폴리오 비중
        notional_usd: 목표 명목 금액 (USD)
        price: 지정가 (LIMIT 주문 시)
        validated: RM 검증 통과 여부
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.ORDER_REQUEST
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    client_order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    order_type: Literal["MARKET", "LIMIT"] = "MARKET"
    target_weight: float
    notional_usd: float
    price: float | None = None
    validated: bool = False


class OrderAckEvent(BaseModel):
    """주문 접수 확인 이벤트.

    OMS가 주문을 접수했음을 알리는 이벤트입니다.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.ORDER_ACK
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    client_order_id: str
    symbol: str


class OrderRejectedEvent(BaseModel):
    """주문 거부 이벤트.

    RM 또는 OMS가 주문을 거부했을 때 발행됩니다.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.ORDER_REJECTED
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    client_order_id: str
    symbol: str
    reason: str


class FillEvent(BaseModel):
    """체결 이벤트.

    Executor가 주문 체결 시 발행합니다.

    Attributes:
        client_order_id: 원본 주문의 멱등성 키
        symbol: 거래 심볼
        side: 매수/매도
        fill_price: 체결 가격
        fill_qty: 체결 수량
        fee: 수수료
        fill_timestamp: 체결 시각
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.FILL
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    client_order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    fill_price: float
    fill_qty: float
    fee: float = 0.0
    fill_timestamp: datetime


# ==========================================================================
# Portfolio Events
# ==========================================================================
class PositionUpdateEvent(BaseModel):
    """포지션 업데이트 이벤트.

    체결 후 PM이 포지션 상태를 발행합니다.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.POSITION_UPDATE
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    symbol: str
    direction: Direction
    size: float
    avg_entry_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class BalanceUpdateEvent(BaseModel):
    """잔고 업데이트 이벤트.

    체결 후 PM이 계좌 잔고를 발행합니다.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.BALANCE_UPDATE
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    total_equity: float
    available_cash: float
    total_margin_used: float = 0.0


# ==========================================================================
# Risk Events
# ==========================================================================
class RiskAlertEvent(BaseModel):
    """리스크 경고 이벤트.

    RiskManager가 임계값 접근 시 발행합니다.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.RISK_ALERT
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    alert_level: Literal["WARNING", "CRITICAL"]
    message: str


class CircuitBreakerEvent(BaseModel):
    """서킷 브레이커 이벤트.

    Kill Switch 발동 시 RiskManager가 발행합니다.
    OMS가 구독하여 전량 청산을 실행합니다.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.CIRCUIT_BREAKER
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    reason: str
    close_all_positions: bool = True


# ==========================================================================
# System Events
# ==========================================================================
class HeartbeatEvent(BaseModel):
    """컴포넌트 헬스체크 이벤트."""

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType = EventType.HEARTBEAT
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: UUID | None = None
    source: str = ""

    component: str
    bars_processed: int = 0


# ==========================================================================
# Union type for all events (dispatch용)
# ==========================================================================
type AnyEvent = (
    BarEvent
    | SignalEvent
    | OrderRequestEvent
    | OrderAckEvent
    | OrderRejectedEvent
    | FillEvent
    | PositionUpdateEvent
    | BalanceUpdateEvent
    | RiskAlertEvent
    | CircuitBreakerEvent
    | HeartbeatEvent
)

EVENT_TYPE_MAP: dict[EventType, type[AnyEvent]] = {
    EventType.BAR: BarEvent,
    EventType.SIGNAL: SignalEvent,
    EventType.ORDER_REQUEST: OrderRequestEvent,
    EventType.ORDER_ACK: OrderAckEvent,
    EventType.ORDER_REJECTED: OrderRejectedEvent,
    EventType.FILL: FillEvent,
    EventType.POSITION_UPDATE: PositionUpdateEvent,
    EventType.BALANCE_UPDATE: BalanceUpdateEvent,
    EventType.RISK_ALERT: RiskAlertEvent,
    EventType.CIRCUIT_BREAKER: CircuitBreakerEvent,
    EventType.HEARTBEAT: HeartbeatEvent,
}

# 절대 드롭 불가 이벤트 (backpressure 정책)
NEVER_DROP_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.SIGNAL,
        EventType.FILL,
        EventType.ORDER_REQUEST,
        EventType.ORDER_ACK,
        EventType.ORDER_REJECTED,
        EventType.POSITION_UPDATE,
        EventType.BALANCE_UPDATE,
        EventType.CIRCUIT_BREAKER,
    }
)

# 드롭 가능 이벤트 (stale 데이터 드롭)
DROPPABLE_EVENTS: frozenset[EventType] = frozenset(
    {
        EventType.BAR,
        EventType.HEARTBEAT,
        EventType.RISK_ALERT,
    }
)
