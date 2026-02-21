"""EDA Order Management System.

validated 주문을 Executor로 라우팅하고, 멱등성을 보장합니다.
CircuitBreaker 이벤트 수신 시 전량 청산을 실행합니다.

흐름: RM → OrderRequest(validated=True) → OMS → Executor → FillEvent

Rules Applied:
    - Idempotency: client_order_id 기반 중복 방지
    - Circuit Breaker: 전량 청산 실행
    - Executor Routing: 백테스트/라이브 실행기 라우팅
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import (
    AnyEvent,
    CircuitBreakerEvent,
    EventType,
    OrderAckEvent,
    OrderRejectedEvent,
    OrderRequestEvent,
)
from src.eda.ports import ExecutorPort
from src.logging.tracing import component_span_with_context

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.eda.portfolio_manager import EDAPortfolioManager

Executor = ExecutorPort
"""Backward-compatible alias for ExecutorPort."""


_MAX_PROCESSED_IN_MEMORY = 100_000


class OMS:
    """Order Management System.

    Subscribes to: OrderRequestEvent(validated=True), CircuitBreakerEvent
    Publishes: OrderAckEvent, FillEvent

    Args:
        executor: 주문 실행기
        portfolio_manager: PM 참조 (청산용)
    """

    def __init__(
        self,
        executor: ExecutorPort,
        portfolio_manager: EDAPortfolioManager | None = None,
    ) -> None:
        self._executor = executor
        self._pm = portfolio_manager
        self._bus: EventBus | None = None
        # dict[str, None]로 삽입 순서 보장 (FIFO eviction용)
        self._processed_orders: dict[str, None] = {}
        self._total_fills = 0
        self._total_rejected = 0

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록."""
        self._bus = bus
        bus.subscribe(EventType.ORDER_REQUEST, self._on_order_request)
        bus.subscribe(EventType.CIRCUIT_BREAKER, self._on_circuit_breaker)

    @property
    def total_fills(self) -> int:
        """총 체결 건수."""
        return self._total_fills

    @property
    def total_rejected(self) -> int:
        """총 거부 건수."""
        return self._total_rejected

    @property
    def processed_orders(self) -> set[str]:
        """처리된 주문 ID 집합 (StateManager 접근용)."""
        return set(self._processed_orders)

    def restore_processed_orders(self, order_ids: set[str]) -> None:
        """저장된 주문 ID를 복원 (재시작 시 중복 방지).

        Args:
            order_ids: StateManager에서 로드한 주문 ID 집합
        """
        self._processed_orders = dict.fromkeys(order_ids)
        logger.info(
            "OMS: restored {} processed order IDs from persistence",
            len(order_ids),
        )

    async def _on_order_request(self, event: AnyEvent) -> None:
        """validated 주문 처리."""
        assert isinstance(event, OrderRequestEvent)
        order = event
        bus = self._bus
        assert bus is not None

        # validated 주문만 처리
        if not order.validated:
            return

        corr_id = str(order.correlation_id) if order.correlation_id else None
        with component_span_with_context("oms.submit_order", corr_id, {"symbol": order.symbol}):
            await self._on_order_request_inner(order, bus)

    async def _on_order_request_inner(self, order: OrderRequestEvent, bus: EventBus) -> None:
        """_on_order_request 본체 (tracing span 내부)."""
        # 멱등성 체크
        if order.client_order_id in self._processed_orders:
            logger.warning("Duplicate order ignored: {}", order.client_order_id)
            self._total_rejected += 1
            rejected = OrderRejectedEvent(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                reason="Duplicate order",
                correlation_id=order.correlation_id,
                source="OMS",
            )
            await bus.publish(rejected)
            return

        self._processed_orders[order.client_order_id] = None
        if len(self._processed_orders) > _MAX_PROCESSED_IN_MEMORY:
            oldest = next(iter(self._processed_orders))
            del self._processed_orders[oldest]

        # OrderAck 발행
        ack = OrderAckEvent(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            correlation_id=order.correlation_id,
            source="OMS",
        )
        await bus.publish(ack)

        # Executor 실행 (None = deferred or shadow, not rejection)
        fill = await self._executor.execute(order)
        if fill is not None:
            self._total_fills += 1
            await bus.publish(fill)

    async def _on_circuit_breaker(self, event: AnyEvent) -> None:
        """CircuitBreaker → 전량 청산."""
        assert isinstance(event, CircuitBreakerEvent)

        if not event.close_all_positions:
            return

        if self._pm is None:
            logger.warning("Circuit breaker: no PM reference for close-all")
            return

        bus = self._bus
        assert bus is not None

        # 모든 오픈 포지션에 대해 청산 주문 생성
        from src.models.types import Direction

        for symbol, pos in self._pm.positions.items():
            if not pos.is_open:
                continue

            side = "SELL" if pos.direction == Direction.LONG else "BUY"
            close_order = OrderRequestEvent(
                client_order_id=f"cb-close-{symbol}",
                symbol=symbol,
                side=side,  # type: ignore[arg-type]
                target_weight=0.0,
                notional_usd=pos.notional,
                price=pos.last_price if pos.last_price > 0 else None,
                validated=True,
                correlation_id=event.correlation_id,
                source="OMS-CircuitBreaker",
            )

            # 직접 실행 (RM 우회)
            fill = await self._executor.execute(close_order)
            if fill is not None:
                self._total_fills += 1
                await bus.publish(fill)

        logger.critical("Circuit breaker: all positions closed")
