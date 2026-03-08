"""NotificationEngine 테스트 -- EventBus 연동."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from src.core.event_bus import EventBus
from src.notification.engine import NotificationEngine
from src.notification.models import ChannelRoute, Severity

if TYPE_CHECKING:
    from src.core.events import (
        BalanceUpdateEvent,
        CircuitBreakerEvent,
        FillEvent,
        PositionUpdateEvent,
        RiskAlertEvent,
    )


class TestNotificationEngine:
    async def test_fill_buffered_not_enqueued(self, sample_fill: FillEvent) -> None:
        """FillEvent 단독 발행 시 버퍼링만 되고 enqueue 안 됨."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_fill)
        await bus.flush()

        mock_queue.enqueue.assert_not_called()
        assert engine._pending_fills["BTC/USDT"] is sample_fill

        await bus.stop()
        await bus_task

    async def test_fill_then_position_combined(
        self,
        sample_fill: FillEvent,
        sample_position_update: PositionUpdateEvent,
    ) -> None:
        """Fill + PositionUpdate가 하나의 통합 알림으로 enqueue됨."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_fill)
        await bus.publish(sample_position_update)
        await bus.flush()

        mock_queue.enqueue.assert_called_once()
        item = mock_queue.enqueue.call_args[0][0]
        assert item.severity == Severity.INFO
        assert item.channel == ChannelRoute.TRADE_LOG
        assert "BUY" in item.embed["title"]
        # 통합 embed에는 Position/Avg Entry/Realized PnL 필드 포함
        field_names = [f["name"] for f in item.embed["fields"]]
        assert "Position" in field_names
        assert "Avg Entry" in field_names
        assert "Realized PnL" in field_names
        # 버퍼 비워짐
        assert "BTC/USDT" not in engine._pending_fills

        await bus.stop()
        await bus_task

    async def test_circuit_breaker_enqueued(
        self, sample_circuit_breaker: CircuitBreakerEvent
    ) -> None:
        """CircuitBreakerEvent가 ALERTS 채널에 CRITICAL로 enqueue됨."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_circuit_breaker)
        await bus.flush()

        item = mock_queue.enqueue.call_args[0][0]
        assert item.severity == Severity.CRITICAL
        assert item.channel == ChannelRoute.ALERTS
        assert "CIRCUIT BREAKER" in item.embed["title"]

        await bus.stop()
        await bus_task

    async def test_risk_alert_warning(self, sample_risk_alert_warning: RiskAlertEvent) -> None:
        """WARNING RiskAlertEvent가 spam_key와 함께 enqueue됨."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_risk_alert_warning)
        await bus.flush()

        item = mock_queue.enqueue.call_args[0][0]
        assert item.severity == Severity.WARNING
        assert item.channel == ChannelRoute.ALERTS
        assert item.spam_key == "risk_alert:WARNING"

        await bus.stop()
        await bus_task

    async def test_risk_alert_critical(self, sample_risk_alert_critical: RiskAlertEvent) -> None:
        """CRITICAL RiskAlertEvent의 severity가 CRITICAL."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_risk_alert_critical)
        await bus.flush()

        item = mock_queue.enqueue.call_args[0][0]
        assert item.severity == Severity.CRITICAL
        assert item.spam_key == "risk_alert:CRITICAL"

        await bus.stop()
        await bus_task

    async def test_balance_update_not_subscribed(
        self, sample_balance_update: BalanceUpdateEvent
    ) -> None:
        """BalanceUpdateEvent는 구독하지 않음 (Bar Close 리포트로 대체)."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_balance_update)
        await bus.flush()

        # Balance Update는 더 이상 enqueue되지 않아야 함
        mock_queue.enqueue.assert_not_called()

        await bus.stop()
        await bus_task

    async def test_register_does_not_subscribe_balance_update(self) -> None:
        """register() 호출 시 BALANCE_UPDATE를 구독하지 않음."""
        from src.core.events import EventType

        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        # BALANCE_UPDATE 핸들러가 등록되지 않았는지 확인
        balance_handlers = bus._handlers.get(EventType.BALANCE_UPDATE, [])
        assert len(balance_handlers) == 0

    async def test_position_update_standalone(
        self, sample_position_update: PositionUpdateEvent
    ) -> None:
        """Fill 없이 PositionUpdateEvent만 오면 기존 position embed 발송."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_position_update)
        await bus.flush()

        item = mock_queue.enqueue.call_args[0][0]
        assert item.channel == ChannelRoute.TRADE_LOG
        assert "Position:" in item.embed["title"]
        assert item.spam_key is None

        await bus.stop()
        await bus_task

    async def test_multiple_events(
        self,
        sample_fill: FillEvent,
        sample_position_update: PositionUpdateEvent,
        sample_circuit_breaker: CircuitBreakerEvent,
    ) -> None:
        """여러 이벤트가 각각 적절히 enqueue됨."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_fill)
        await bus.publish(sample_position_update)
        await bus.publish(sample_circuit_breaker)
        await bus.flush()

        # Fill+Position 통합 1건 + CircuitBreaker 1건 = 2건
        assert mock_queue.enqueue.call_count == 2

        await bus.stop()
        await bus_task
