"""OMS processed_orders bounded dict 테스트.

FIFO eviction과 멱등성이 올바르게 동작하는지 검증합니다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.events import OrderRequestEvent
from src.eda.oms import _MAX_PROCESSED_IN_MEMORY, OMS


class _FakeExecutor:
    """테스트용 Executor stub."""

    async def execute(self, order: OrderRequestEvent) -> None:
        return None


class TestOMSBoundedProcessedOrders:
    """OMS dict-based FIFO eviction 검증."""

    def test_processed_orders_returns_set(self) -> None:
        """processed_orders property는 set을 반환한다."""
        oms = OMS(executor=_FakeExecutor())  # type: ignore[arg-type]
        assert isinstance(oms.processed_orders, set)
        assert len(oms.processed_orders) == 0

    def test_restore_converts_set_to_dict(self) -> None:
        """restore_processed_orders는 set을 dict로 변환한다."""
        oms = OMS(executor=_FakeExecutor())  # type: ignore[arg-type]
        oms.restore_processed_orders({"order-1", "order-2", "order-3"})

        assert "order-1" in oms._processed_orders
        assert "order-2" in oms._processed_orders
        assert "order-3" in oms._processed_orders
        assert len(oms._processed_orders) == 3

    @pytest.mark.asyncio
    async def test_fifo_eviction(self) -> None:
        """_MAX_PROCESSED_IN_MEMORY 초과 시 가장 오래된 항목이 제거된다."""
        oms = OMS(executor=_FakeExecutor())  # type: ignore[arg-type]
        bus = AsyncMock()
        bus.publish = AsyncMock()
        oms._bus = bus

        # dict에 직접 MAX - 1개 삽입
        limit = _MAX_PROCESSED_IN_MEMORY
        for i in range(limit):
            oms._processed_orders[f"order-{i}"] = None

        assert len(oms._processed_orders) == limit
        assert "order-0" in oms._processed_orders

        # 한 개 추가 → eviction 발생
        order = OrderRequestEvent(
            client_order_id="new-order",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.1,
            notional_usd=1000.0,
            validated=True,
            source="test",
        )
        await oms._on_order_request_inner(order, bus)

        assert len(oms._processed_orders) == limit
        # 가장 오래된 order-0이 제거됨
        assert "order-0" not in oms._processed_orders
        # 새 주문은 존재
        assert "new-order" in oms._processed_orders
        # order-1은 아직 존재
        assert "order-1" in oms._processed_orders

    @pytest.mark.asyncio
    async def test_idempotency_preserved(self) -> None:
        """FIFO eviction에도 불구하고 최근 주문의 멱등성이 보장된다."""
        oms = OMS(executor=_FakeExecutor())  # type: ignore[arg-type]
        bus = AsyncMock()
        bus.publish = AsyncMock()
        oms._bus = bus

        order = OrderRequestEvent(
            client_order_id="dup-order",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.1,
            notional_usd=1000.0,
            validated=True,
            source="test",
        )

        # 첫 번째 처리
        await oms._on_order_request_inner(order, bus)
        assert oms._total_rejected == 0

        # 동일 주문 재처리 → rejected
        await oms._on_order_request_inner(order, bus)
        assert oms._total_rejected == 1

    @pytest.mark.asyncio
    async def test_evicted_order_not_detected_as_duplicate(self) -> None:
        """eviction된 주문은 중복 감지에서 제외된다 (의도적 동작)."""
        oms = OMS(executor=_FakeExecutor())  # type: ignore[arg-type]
        bus = AsyncMock()
        bus.publish = AsyncMock()
        oms._bus = bus

        # limit까지 채우기
        limit = _MAX_PROCESSED_IN_MEMORY
        for i in range(limit):
            oms._processed_orders[f"order-{i}"] = None

        # order-0 확인 → 현재 존재
        assert "order-0" in oms._processed_orders

        # 새 주문 추가 → order-0 eviction
        new_order = OrderRequestEvent(
            client_order_id="trigger-evict",
            symbol="BTC/USDT",
            side="BUY",
            target_weight=0.1,
            notional_usd=1000.0,
            validated=True,
            source="test",
        )
        await oms._on_order_request_inner(new_order, bus)

        # order-0은 evicted → 더 이상 중복으로 감지 안 됨
        assert "order-0" not in oms._processed_orders

    def test_max_processed_constant(self) -> None:
        """_MAX_PROCESSED_IN_MEMORY 상수 값 검증."""
        assert _MAX_PROCESSED_IN_MEMORY == 100_000

    def test_processed_orders_property_returns_copy(self) -> None:
        """processed_orders property는 내부 dict의 복사본 set을 반환한다."""
        oms = OMS(executor=_FakeExecutor())  # type: ignore[arg-type]
        oms._processed_orders["order-1"] = None

        result = oms.processed_orders
        result.add("order-external")

        # 내부 상태에 영향 없음
        assert "order-external" not in oms._processed_orders
