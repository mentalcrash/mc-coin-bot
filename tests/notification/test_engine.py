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
    async def test_fill_event_enqueued(self, sample_fill: FillEvent) -> None:
        """FillEvent가 TRADE_LOG 채널에 INFO로 enqueue됨."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_fill)
        await bus.flush()

        mock_queue.enqueue.assert_called_once()
        item = mock_queue.enqueue.call_args[0][0]
        assert item.severity == Severity.INFO
        assert item.channel == ChannelRoute.TRADE_LOG
        assert "BUY" in item.embed["title"]
        assert item.spam_key is None

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

    async def test_risk_alert_warning(
        self, sample_risk_alert_warning: RiskAlertEvent
    ) -> None:
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

    async def test_risk_alert_critical(
        self, sample_risk_alert_critical: RiskAlertEvent
    ) -> None:
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

    async def test_balance_update_throttled(
        self, sample_balance_update: BalanceUpdateEvent
    ) -> None:
        """BalanceUpdateEvent에 spam_key가 설정됨."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_balance_update)
        await bus.flush()

        item = mock_queue.enqueue.call_args[0][0]
        assert item.spam_key == "balance_update"
        assert item.channel == ChannelRoute.TRADE_LOG

        await bus.stop()
        await bus_task

    async def test_position_update_enqueued(
        self, sample_position_update: PositionUpdateEvent
    ) -> None:
        """PositionUpdateEvent가 TRADE_LOG에 enqueue됨."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_position_update)
        await bus.flush()

        item = mock_queue.enqueue.call_args[0][0]
        assert item.channel == ChannelRoute.TRADE_LOG
        assert "BTC/USDT" in item.embed["title"]
        assert item.spam_key is None

        await bus.stop()
        await bus_task

    async def test_multiple_events(
        self,
        sample_fill: FillEvent,
        sample_circuit_breaker: CircuitBreakerEvent,
    ) -> None:
        """여러 이벤트가 각각 적절히 enqueue됨."""
        mock_queue = AsyncMock()
        engine = NotificationEngine(mock_queue)

        bus = EventBus(queue_size=100)
        await engine.register(bus)

        bus_task = asyncio.create_task(bus.start())
        await bus.publish(sample_fill)
        await bus.publish(sample_circuit_breaker)
        await bus.flush()

        assert mock_queue.enqueue.call_count == 2

        await bus.stop()
        await bus_task
