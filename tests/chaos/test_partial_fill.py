"""Chaos Test -- Partial fill 시나리오.

부분 체결 시 포트폴리오가 올바르게 처리하는지 검증합니다.
"""

from __future__ import annotations

import pytest

from tests.chaos.conftest import FaultyExecutor, make_order_request

pytestmark = pytest.mark.chaos


class TestPartialFill:
    """Partial fill 시나리오 테스트."""

    async def test_partial_fill_60_percent(self) -> None:
        """60% 부분 체결 시 fill_qty가 정확히 반영."""
        executor = FaultyExecutor(partial_fill_ratio=0.6)
        order = make_order_request(notional_usd=10000.0)

        fill = await executor.execute(order)

        assert fill is not None
        # 60% of 10000 = 6000 notional → qty = 6000 / 50000
        expected_qty = 6000.0 / 50000.0
        assert fill.fill_qty == pytest.approx(expected_qty, rel=0.01)

    async def test_full_fill_without_ratio(self) -> None:
        """partial_fill_ratio=None → 전량 체결."""
        executor = FaultyExecutor(partial_fill_ratio=None)
        order = make_order_request(notional_usd=10000.0)

        fill = await executor.execute(order)

        assert fill is not None
        expected_qty = 10000.0 / 50000.0
        assert fill.fill_qty == pytest.approx(expected_qty, rel=0.01)

    async def test_multiple_partial_fills(self) -> None:
        """연속 부분 체결이 독립적으로 처리."""
        executor = FaultyExecutor(partial_fill_ratio=0.5)

        for i in range(5):
            order = make_order_request(
                symbol=f"ASSET{i}/USDT",
                notional_usd=10000.0,
                client_order_id=f"order-{i}",
            )
            fill = await executor.execute(order)
            assert fill is not None
            assert fill.fill_qty == pytest.approx(5000.0 / 50000.0, rel=0.01)

        assert executor.order_count == 5
        assert len(executor.fills) == 5

    async def test_partial_fill_preserves_order_id(self) -> None:
        """부분 체결 시 client_order_id 보존."""
        executor = FaultyExecutor(partial_fill_ratio=0.3)
        order = make_order_request(client_order_id="unique-id-123")

        fill = await executor.execute(order)

        assert fill is not None
        assert fill.client_order_id == "unique-id-123"

    async def test_partial_fill_fee_proportional(self) -> None:
        """부분 체결 시 수수료도 비례 적용."""
        executor = FaultyExecutor(partial_fill_ratio=0.5)
        order = make_order_request(notional_usd=10000.0)

        fill = await executor.execute(order)

        assert fill is not None
        # fee = 5000 * 0.0004 = 2.0
        assert fill.fee == pytest.approx(2.0, abs=0.01)
