"""Chaos Test -- Order timeout + 멱등성 검증.

타임아웃/에러 발생 시 주문 처리가 안전한지 검증합니다.
"""

from __future__ import annotations

import pytest

from tests.chaos.conftest import FaultyExecutor, make_order_request

pytestmark = pytest.mark.chaos


class TestOrderTimeout:
    """주문 타임아웃 시나리오 테스트."""

    async def test_timeout_returns_none(self) -> None:
        """타임아웃 시 None 반환 (fill 없음)."""
        executor = FaultyExecutor(timeout_probability=1.0)
        order = make_order_request()

        fill = await executor.execute(order)

        assert fill is None
        assert len(executor.timeouts) == 1
        assert len(executor.fills) == 0

    async def test_mixed_timeout_and_success(self) -> None:
        """일부 타임아웃, 일부 성공 혼합."""
        executor = FaultyExecutor(timeout_probability=0.5, seed=42)

        fills = 0
        timeouts = 0
        for i in range(20):
            order = make_order_request(client_order_id=f"order-{i}")
            fill = await executor.execute(order)
            if fill is not None:
                fills += 1
            else:
                timeouts += 1

        assert fills > 0
        assert timeouts > 0
        assert fills + timeouts == 20

    async def test_no_timeout_with_zero_probability(self) -> None:
        """timeout_probability=0.0 → 모두 성공."""
        executor = FaultyExecutor(timeout_probability=0.0)

        for i in range(10):
            order = make_order_request(client_order_id=f"order-{i}")
            fill = await executor.execute(order)
            assert fill is not None

        assert executor.order_count == 10
        assert len(executor.fills) == 10


class TestOrderFailure:
    """주문 에러 시나리오 테스트."""

    async def test_fail_after_n(self) -> None:
        """N번째 이후 주문에서 에러 발생."""
        executor = FaultyExecutor(fail_after_n=3)

        # 처음 3번은 성공
        for i in range(3):
            order = make_order_request(client_order_id=f"order-{i}")
            fill = await executor.execute(order)
            assert fill is not None

        # 4번째에서 에러
        order = make_order_request(client_order_id="order-3")
        with pytest.raises(RuntimeError, match="Simulated failure"):
            await executor.execute(order)

        assert len(executor.errors) == 1

    async def test_error_preserves_order_state(self) -> None:
        """에러 발생해도 이전 fills은 보존."""
        executor = FaultyExecutor(fail_after_n=2)

        order1 = make_order_request(client_order_id="order-1")
        order2 = make_order_request(client_order_id="order-2")
        await executor.execute(order1)
        await executor.execute(order2)

        assert len(executor.fills) == 2

        # 에러 발생
        order3 = make_order_request(client_order_id="order-3")
        with pytest.raises(RuntimeError):
            await executor.execute(order3)

        # 이전 fills 보존
        assert len(executor.fills) == 2
        assert executor.fills[0].client_order_id == "order-1"


class TestIdempotency:
    """주문 멱등성 테스트."""

    async def test_same_order_id_different_executions(self) -> None:
        """동일 client_order_id로 여러번 실행해도 각각 독립 처리."""
        executor = FaultyExecutor()

        for _ in range(3):
            order = make_order_request(
                client_order_id="same-id",
                notional_usd=10000.0,
            )
            fill = await executor.execute(order)
            assert fill is not None
            assert fill.client_order_id == "same-id"

        # FaultyExecutor는 멱등 보장 안 함 (의도적) — 3개 fill 생성
        assert len(executor.fills) == 3

    async def test_order_tracking_accuracy(self) -> None:
        """주문 카운터 정확성."""
        executor = FaultyExecutor()

        for i in range(7):
            order = make_order_request(client_order_id=f"ord-{i}")
            await executor.execute(order)

        assert executor.order_count == 7
