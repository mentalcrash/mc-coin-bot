"""NotificationEngine -- EventBus subscriber -> NotificationQueue.

TradePersistence와 동일한 패턴으로 EventBus 이벤트를 구독하여
Discord 알림으로 변환합니다. fire-and-forget으로 거래 로직을 차단하지 않습니다.

Rules Applied:
    - EDA 패턴: EventBus subscribe, fire-and-forget
    - #10 Python Standards: Async patterns, type hints
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    CircuitBreakerEvent,
    EventType,
    FillEvent,
    PositionUpdateEvent,
    RiskAlertEvent,
)
from src.notification.formatters import (
    format_balance_embed,
    format_circuit_breaker_embed,
    format_fill_with_position_embed,
    format_position_embed,
    format_risk_alert_embed,
)
from src.notification.models import ChannelRoute, NotificationItem, Severity

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.notification.queue import NotificationQueue


class NotificationEngine:
    """EventBus subscriber -- 이벤트를 알림으로 변환하여 큐에 전달.

    Args:
        queue: NotificationQueue 인스턴스
    """

    def __init__(self, queue: NotificationQueue) -> None:
        self._queue = queue
        self._pending_fills: dict[str, FillEvent] = {}

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록.

        Args:
            bus: EventBus 인스턴스
        """
        bus.subscribe(EventType.FILL, self._on_fill)
        bus.subscribe(EventType.CIRCUIT_BREAKER, self._on_circuit_breaker)
        bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert)
        bus.subscribe(EventType.BALANCE_UPDATE, self._on_balance_update)
        bus.subscribe(EventType.POSITION_UPDATE, self._on_position_update)
        logger.info("NotificationEngine registered to EventBus")

    async def _on_fill(self, event: AnyEvent) -> None:
        """FillEvent -> 버퍼링 후 PositionUpdateEvent와 통합 발송."""
        assert isinstance(event, FillEvent)
        self._pending_fills[event.symbol] = event

    async def _on_circuit_breaker(self, event: AnyEvent) -> None:
        """CircuitBreakerEvent -> ALERTS 채널 (CRITICAL)."""
        assert isinstance(event, CircuitBreakerEvent)
        embed = format_circuit_breaker_embed(event)
        item = NotificationItem(
            severity=Severity.CRITICAL,
            channel=ChannelRoute.ALERTS,
            embed=embed,
        )
        await self._queue.enqueue(item)

    async def _on_risk_alert(self, event: AnyEvent) -> None:
        """RiskAlertEvent -> ALERTS 채널."""
        assert isinstance(event, RiskAlertEvent)
        embed = format_risk_alert_embed(event)
        severity = Severity.CRITICAL if event.alert_level == "CRITICAL" else Severity.WARNING
        item = NotificationItem(
            severity=severity,
            channel=ChannelRoute.ALERTS,
            embed=embed,
            spam_key=f"risk_alert:{event.alert_level}",
        )
        await self._queue.enqueue(item)

    async def _on_balance_update(self, event: AnyEvent) -> None:
        """BalanceUpdateEvent -> TRADE_LOG 채널 (throttled)."""
        assert isinstance(event, BalanceUpdateEvent)
        embed = format_balance_embed(event)
        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.TRADE_LOG,
            embed=embed,
            spam_key="balance_update",
        )
        await self._queue.enqueue(item)

    async def _on_position_update(self, event: AnyEvent) -> None:
        """PositionUpdateEvent -> Fill과 통합 또는 단독 발송."""
        assert isinstance(event, PositionUpdateEvent)
        pending_fill = self._pending_fills.pop(event.symbol, None)
        if pending_fill is not None:
            embed = format_fill_with_position_embed(pending_fill, event)
        else:
            embed = format_position_embed(event)
        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.TRADE_LOG,
            embed=embed,
        )
        await self._queue.enqueue(item)
