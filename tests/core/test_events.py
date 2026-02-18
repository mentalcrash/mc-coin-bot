"""이벤트 타입 계층 테스트.

BaseEvent 및 모든 구체 이벤트 타입의 생성, 불변성, 직렬화를 검증합니다.
"""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from src.core.events import (
    DROPPABLE_EVENTS,
    EVENT_TYPE_MAP,
    NEVER_DROP_EVENTS,
    BalanceUpdateEvent,
    BarEvent,
    CircuitBreakerEvent,
    EventType,
    FillEvent,
    HeartbeatEvent,
    OrderAckEvent,
    OrderRejectedEvent,
    OrderRequestEvent,
    PositionUpdateEvent,
    RiskAlertEvent,
    SignalEvent,
)
from src.models.types import Direction


class TestEventType:
    """EventType StrEnum 테스트."""

    def test_all_event_types_defined(self) -> None:
        assert len(EventType) == 12

    def test_event_type_values(self) -> None:
        assert EventType.BAR == "bar"
        assert EventType.SIGNAL == "signal"
        assert EventType.FILL == "fill"
        assert EventType.CIRCUIT_BREAKER == "circuit_breaker"

    def test_event_type_map_covers_all(self) -> None:
        """EVENT_TYPE_MAP이 모든 EventType을 포함하는지 검증."""
        for et in EventType:
            assert et in EVENT_TYPE_MAP, f"{et} not in EVENT_TYPE_MAP"


class TestCommonEventFields:
    """공통 이벤트 필드 테스트 (BarEvent 사용)."""

    def test_auto_generated_fields(self) -> None:
        now = datetime.now(UTC)
        event = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            bar_timestamp=now,
        )
        assert isinstance(event.event_id, UUID)
        assert event.event_type == EventType.BAR
        assert event.timestamp.tzinfo is not None
        assert event.correlation_id is None
        assert event.source == ""

    def test_frozen_immutability(self) -> None:
        now = datetime.now(UTC)
        event = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            bar_timestamp=now,
        )
        with pytest.raises(ValidationError):
            event.source = "modified"  # type: ignore[misc]

    def test_custom_correlation_id(self) -> None:
        cid = uuid4()
        now = datetime.now(UTC)
        event = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            bar_timestamp=now,
            correlation_id=cid,
            source="TestComponent",
        )
        assert event.correlation_id == cid
        assert event.source == "TestComponent"


class TestBarEvent:
    """BarEvent 테스트."""

    def test_create(self) -> None:
        now = datetime.now(UTC)
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            bar_timestamp=now,
        )
        assert bar.event_type == EventType.BAR
        assert bar.symbol == "BTC/USDT"
        assert bar.close == 50500.0

    def test_event_type_is_literal(self) -> None:
        """event_type이 자동으로 BAR로 설정되는지 확인."""
        now = datetime.now(UTC)
        bar = BarEvent(
            symbol="ETH/USDT",
            timeframe="4h",
            open=3000.0,
            high=3100.0,
            low=2900.0,
            close=3050.0,
            volume=500.0,
            bar_timestamp=now,
        )
        assert bar.event_type == EventType.BAR

    def test_json_serialization(self) -> None:
        now = datetime.now(UTC)
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=1000.0,
            bar_timestamp=now,
        )
        json_str = bar.model_dump_json()
        assert "BTC/USDT" in json_str
        assert "50500" in json_str


class TestSignalEvent:
    """SignalEvent 테스트."""

    def test_create_long_signal(self) -> None:
        now = datetime.now(UTC)
        signal = SignalEvent(
            symbol="BTC/USDT",
            strategy_name="tsmom",
            direction=Direction.LONG,
            strength=1.5,
            bar_timestamp=now,
            source="StrategyEngine",
        )
        assert signal.event_type == EventType.SIGNAL
        assert signal.direction == Direction.LONG
        assert signal.strength == 1.5

    def test_create_short_signal(self) -> None:
        now = datetime.now(UTC)
        signal = SignalEvent(
            symbol="ETH/USDT",
            strategy_name="tsmom",
            direction=Direction.SHORT,
            strength=-0.5,
            bar_timestamp=now,
        )
        assert signal.direction == Direction.SHORT

    def test_correlation_id_propagation(self) -> None:
        """BarEvent의 correlation_id가 SignalEvent로 전파되는지 확인."""
        cid = uuid4()
        now = datetime.now(UTC)
        signal = SignalEvent(
            symbol="BTC/USDT",
            strategy_name="tsmom",
            direction=Direction.LONG,
            strength=1.0,
            bar_timestamp=now,
            correlation_id=cid,
        )
        assert signal.correlation_id == cid


class TestOrderRequestEvent:
    """OrderRequestEvent 테스트."""

    def test_create_buy_order(self) -> None:
        order = OrderRequestEvent(
            client_order_id="tsmom-BTC/USDT-1",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        assert order.event_type == EventType.ORDER_REQUEST
        assert order.validated is False
        assert order.order_type == "MARKET"

    def test_validated_flag(self) -> None:
        order = OrderRequestEvent(
            client_order_id="tsmom-BTC/USDT-2",
            symbol="BTC/USDT",
            side="SELL",
            target_weight=-0.3,
            notional_usd=3000.0,
            validated=True,
        )
        assert order.validated is True

    def test_model_copy_update(self) -> None:
        """RM이 validated=True로 업데이트하는 패턴 검증."""
        order = OrderRequestEvent(
            client_order_id="tsmom-BTC/USDT-3",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        validated = order.model_copy(update={"validated": True})
        assert validated.validated is True
        assert validated.client_order_id == order.client_order_id
        assert order.validated is False  # 원본 불변


class TestFillEvent:
    """FillEvent 테스트."""

    def test_create(self) -> None:
        now = datetime.now(UTC)
        fill = FillEvent(
            client_order_id="tsmom-BTC/USDT-1",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50000.0,
            fill_qty=0.1,
            fee=2.2,
            fill_timestamp=now,
        )
        assert fill.event_type == EventType.FILL
        assert fill.fill_price == 50000.0
        assert fill.fee == 2.2

    def test_default_fee(self) -> None:
        now = datetime.now(UTC)
        fill = FillEvent(
            client_order_id="test-1",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50000.0,
            fill_qty=0.1,
            fill_timestamp=now,
        )
        assert fill.fee == 0.0


class TestPortfolioEvents:
    """PositionUpdateEvent, BalanceUpdateEvent 테스트."""

    def test_position_update(self) -> None:
        pos = PositionUpdateEvent(
            symbol="BTC/USDT",
            direction=Direction.LONG,
            size=0.5,
            avg_entry_price=50000.0,
            unrealized_pnl=500.0,
        )
        assert pos.event_type == EventType.POSITION_UPDATE
        assert pos.direction == Direction.LONG

    def test_balance_update(self) -> None:
        bal = BalanceUpdateEvent(
            total_equity=10500.0,
            available_cash=5000.0,
            total_margin_used=5500.0,
        )
        assert bal.event_type == EventType.BALANCE_UPDATE
        assert bal.total_equity == 10500.0


class TestRiskEvents:
    """RiskAlertEvent, CircuitBreakerEvent 테스트."""

    def test_risk_alert(self) -> None:
        alert = RiskAlertEvent(
            alert_level="WARNING",
            message="Leverage approaching limit: 1.8x / 2.0x",
        )
        assert alert.event_type == EventType.RISK_ALERT
        assert alert.alert_level == "WARNING"

    def test_circuit_breaker(self) -> None:
        cb = CircuitBreakerEvent(
            reason="System stop-loss triggered: 10.5% drawdown",
        )
        assert cb.event_type == EventType.CIRCUIT_BREAKER
        assert cb.close_all_positions is True


class TestSystemEvents:
    """HeartbeatEvent 테스트."""

    def test_heartbeat(self) -> None:
        hb = HeartbeatEvent(component="DataFeed", bars_processed=150)
        assert hb.event_type == EventType.HEARTBEAT
        assert hb.bars_processed == 150


class TestOrderAckAndRejected:
    """OrderAckEvent, OrderRejectedEvent 테스트."""

    def test_order_ack(self) -> None:
        ack = OrderAckEvent(
            client_order_id="tsmom-BTC/USDT-1",
            symbol="BTC/USDT",
        )
        assert ack.event_type == EventType.ORDER_ACK

    def test_order_rejected(self) -> None:
        rejected = OrderRejectedEvent(
            client_order_id="tsmom-BTC/USDT-1",
            symbol="BTC/USDT",
            reason="Aggregate leverage exceeded: 2.5x > 2.0x",
        )
        assert rejected.event_type == EventType.ORDER_REJECTED
        assert "2.5x" in rejected.reason


class TestBackpressurePolicy:
    """드롭 정책 상수 테스트."""

    def test_never_drop_and_droppable_are_disjoint(self) -> None:
        assert NEVER_DROP_EVENTS.isdisjoint(DROPPABLE_EVENTS)

    def test_all_event_types_classified(self) -> None:
        """모든 EventType이 NEVER_DROP 또는 DROPPABLE에 분류되는지 확인."""
        all_classified = NEVER_DROP_EVENTS | DROPPABLE_EVENTS
        for et in EventType:
            assert et in all_classified, f"{et} not classified in backpressure policy"

    def test_critical_events_never_dropped(self) -> None:
        assert EventType.SIGNAL in NEVER_DROP_EVENTS
        assert EventType.FILL in NEVER_DROP_EVENTS
        assert EventType.CIRCUIT_BREAKER in NEVER_DROP_EVENTS

    def test_market_data_droppable(self) -> None:
        assert EventType.BAR in DROPPABLE_EVENTS
        assert EventType.HEARTBEAT in DROPPABLE_EVENTS
