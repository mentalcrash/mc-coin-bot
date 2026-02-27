"""SmartExecutor 테스트.

Limit order 우선 실행기의 urgency 분류, limit lifecycle, fallback, merge 등을 검증합니다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.core.events import FillEvent, OrderRequestEvent
from src.eda.smart_executor import SmartExecutor, Urgency
from src.eda.smart_executor_config import SmartExecutorConfig


def _make_order(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    notional_usd: float = 10000.0,
    price: float | None = None,
    target_weight: float = 0.5,
) -> OrderRequestEvent:
    return OrderRequestEvent(
        client_order_id=f"test-{uuid4().hex[:8]}",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        target_weight=target_weight,
        notional_usd=notional_usd,
        price=price,
        validated=True,
        correlation_id=uuid4(),
        source="test",
    )


def _make_fill(
    order: OrderRequestEvent,
    fill_price: float = 50000.0,
    fill_qty: float = 0.2,
    fee: float = 2.0,
    source: str = "LiveExecutor",
) -> FillEvent:
    return FillEvent(
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        side=order.side,
        fill_price=fill_price,
        fill_qty=fill_qty,
        fee=fee,
        fill_timestamp=datetime.now(UTC),
        correlation_id=order.correlation_id,
        source=source,
    )


def _make_smart_executor(
    *,
    enabled: bool = True,
    api_healthy: bool = True,
    limit_timeout_seconds: float = 30.0,
    max_concurrent: int = 4,
) -> tuple[SmartExecutor, MagicMock, MagicMock]:
    """SmartExecutor + mock inner + mock client 생성."""
    mock_inner = MagicMock()
    mock_inner.execute = AsyncMock(return_value=None)
    mock_inner._pm = None

    mock_client = MagicMock()
    mock_client.is_api_healthy = api_healthy
    mock_client.consecutive_failures = 0
    mock_client.to_futures_symbol = MagicMock(return_value="BTC/USDT:USDT")
    mock_client.fetch_ticker = AsyncMock(return_value={"bid": 50000.0, "ask": 50010.0})
    mock_client.create_order = AsyncMock(return_value={"id": "order123", "status": "open"})
    mock_client.fetch_order = AsyncMock(
        return_value={"id": "order123", "status": "closed", "filled": 0.2, "average": 50005.0, "fee": {"cost": 2.0}}
    )
    mock_client.cancel_order = AsyncMock(return_value={"id": "order123", "status": "canceled"})
    mock_client.fetch_open_orders = AsyncMock(return_value=[])

    config = SmartExecutorConfig(
        enabled=enabled,
        limit_timeout_seconds=limit_timeout_seconds,
        max_concurrent_limit_orders=max_concurrent,
    )

    executor = SmartExecutor(inner=mock_inner, config=config, futures_client=mock_client)
    return executor, mock_inner, mock_client


class TestUrgencyClassification:
    """Urgency 분류 테스트."""

    def test_disabled_config_returns_urgent(self) -> None:
        """enabled=False → URGENT."""
        executor, _, _ = _make_smart_executor(enabled=False)
        order = _make_order()
        assert executor._classify_urgency(order) == Urgency.URGENT

    def test_sl_order_returns_urgent(self) -> None:
        """SL/TS (price 설정) → URGENT."""
        executor, _, _ = _make_smart_executor()
        order = _make_order(price=47000.0)
        assert executor._classify_urgency(order) == Urgency.URGENT

    def test_flat_close_returns_urgent(self) -> None:
        """target_weight=0 → URGENT."""
        executor, _, _ = _make_smart_executor()
        order = _make_order(target_weight=0)
        assert executor._classify_urgency(order) == Urgency.URGENT

    def test_api_unhealthy_returns_urgent(self) -> None:
        """API unhealthy → URGENT."""
        executor, _, _ = _make_smart_executor(api_healthy=False)
        order = _make_order()
        assert executor._classify_urgency(order) == Urgency.URGENT

    def test_max_concurrent_exceeded_returns_urgent(self) -> None:
        """동시 limit 초과 → URGENT."""
        executor, _, _ = _make_smart_executor(max_concurrent=2)
        executor._active_limit_count = 2
        order = _make_order()
        assert executor._classify_urgency(order) == Urgency.URGENT

    def test_normal_entry_returns_normal(self) -> None:
        """일반 진입 → NORMAL."""
        executor, _, _ = _make_smart_executor()
        order = _make_order()
        assert executor._classify_urgency(order) == Urgency.NORMAL

    def test_direction_flip_returns_urgent(self) -> None:
        """방향 전환 → URGENT."""
        from src.models.types import Direction

        executor, mock_inner, _ = _make_smart_executor()

        mock_pm = MagicMock()
        mock_pos = MagicMock()
        mock_pos.is_open = True
        mock_pos.direction = Direction.SHORT
        mock_pm.positions = {"BTC/USDT": mock_pos}
        mock_inner._pm = mock_pm

        # BUY with current SHORT → direction flip
        order = _make_order(side="BUY", target_weight=0.5)
        assert executor._classify_urgency(order) == Urgency.URGENT


class TestSmartExecutorDelegation:
    """SmartExecutor → inner 위임 테스트."""

    async def test_urgent_sl_delegates_market(self) -> None:
        """SL/TS → inner.execute() 직행."""
        executor, mock_inner, _ = _make_smart_executor()
        expected_fill = _make_fill(_make_order())
        mock_inner.execute = AsyncMock(return_value=expected_fill)

        order = _make_order(price=47000.0)
        result = await executor.execute(order)
        mock_inner.execute.assert_called_once_with(order)
        assert result == expected_fill

    async def test_urgent_flat_close_market(self) -> None:
        """target_weight=0 → market."""
        executor, mock_inner, _ = _make_smart_executor()
        mock_inner.execute = AsyncMock(return_value=None)

        order = _make_order(target_weight=0)
        await executor.execute(order)
        mock_inner.execute.assert_called_once_with(order)

    async def test_disabled_config_passthrough(self) -> None:
        """enabled=False → 전부 market."""
        executor, mock_inner, _ = _make_smart_executor(enabled=False)
        mock_inner.execute = AsyncMock(return_value=None)

        order = _make_order()
        await executor.execute(order)
        mock_inner.execute.assert_called_once_with(order)


class TestLimitOrderLifecycle:
    """Limit 주문 라이프사이클 테스트."""

    async def test_normal_entry_places_limit(self) -> None:
        """일반 진입 → limit 배치 + fetch_order로 확인."""
        executor, _, mock_client = _make_smart_executor()

        order = _make_order()
        result = await executor.execute(order)

        mock_client.create_order.assert_called_once()
        call_kwargs = mock_client.create_order.call_args
        assert call_kwargs.kwargs.get("price") is not None or (
            len(call_kwargs.args) > 3 and call_kwargs.args[3] is not None
        )
        assert result is not None
        assert result.source == "SmartExecutor:limit"

    async def test_limit_full_fill(self) -> None:
        """완전 체결 → FillEvent 반환."""
        executor, _, mock_client = _make_smart_executor()

        mock_client.fetch_order = AsyncMock(return_value={
            "id": "order123",
            "status": "closed",
            "filled": 0.2,
            "average": 50005.0,
            "fee": {"cost": 2.0},
        })

        order = _make_order()
        result = await executor.execute(order)

        assert result is not None
        assert result.fill_price == 50005.0
        assert result.fill_qty == 0.2
        assert result.fee == 2.0

    async def test_limit_timeout_market_fallback(self) -> None:
        """미체결 timeout → cancel → market fallback."""
        executor, mock_inner, mock_client = _make_smart_executor()

        # fetch_order: first poll open, then after cancel: canceled
        poll_count = 0

        async def mock_fetch_order(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal poll_count
            poll_count += 1
            if poll_count <= 1:
                return {"id": "order123", "status": "open", "filled": 0, "average": 0}
            return {"id": "order123", "status": "canceled", "filled": 0, "average": 0}

        mock_client.fetch_order = AsyncMock(side_effect=mock_fetch_order)
        mock_client.fetch_ticker = AsyncMock(return_value={"bid": 50000.0, "ask": 50010.0})

        expected_fill = _make_fill(_make_order())
        mock_inner.execute = AsyncMock(return_value=expected_fill)

        order = _make_order()

        # Patch asyncio.sleep to skip real waiting and mock loop time for deadline
        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("asyncio.get_event_loop") as mock_loop:
            mock_time = MagicMock()
            # First time() call: state.placed_at, then loop time past deadline
            mock_time.time = MagicMock(side_effect=[0.0, 0.0, 100.0])
            mock_loop.return_value = mock_time

            result = await executor.execute(order)

        assert mock_inner.execute.called
        assert result == expected_fill

    async def test_price_deviation_early_cancel(self) -> None:
        """0.3% 이탈 → 즉시 취소."""
        executor, mock_inner, mock_client = _make_smart_executor()

        # First poll: still open
        poll_count = 0

        async def mock_fetch_order(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal poll_count
            poll_count += 1
            if poll_count <= 2:
                return {"id": "order123", "status": "open", "filled": 0, "average": 0}
            # After cancel
            return {"id": "order123", "status": "canceled", "filled": 0, "average": 0}

        mock_client.fetch_order = AsyncMock(side_effect=mock_fetch_order)

        # Ticker deviates significantly (>0.3%) on deviation check
        ticker_count = 0

        async def mock_deviated_ticker(*args: object, **kwargs: object) -> dict[str, float]:
            nonlocal ticker_count
            ticker_count += 1
            if ticker_count <= 1:
                return {"bid": 50000.0, "ask": 50010.0}  # initial for price calc
            # Deviated by >0.3%
            return {"bid": 50200.0, "ask": 50210.0}

        mock_client.fetch_ticker = AsyncMock(side_effect=mock_deviated_ticker)

        expected_fill = _make_fill(_make_order())
        mock_inner.execute = AsyncMock(return_value=expected_fill)

        order = _make_order()

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await executor.execute(order)

        # Should cancel and fallback
        assert mock_client.cancel_order.called

    async def test_cancel_race_with_fill(self) -> None:
        """취소 중 체결 완료 → FillEvent 정상 반환."""
        executor, _, mock_client = _make_smart_executor()

        # _cancel_and_handle_remainder를 직접 테스트 (cancel race 시나리오)
        order = _make_order()
        state = MagicMock()
        state.order = order
        state.exchange_order_id = "order123"
        state.futures_symbol = "BTC/USDT:USDT"
        state.limit_price = 50005.0
        state.reference_price = 50005.0
        state.filled_qty = 0.0
        state.filled_notional = 0.0

        # cancel 실패해도 fetch_order에서 closed 반환 (race)
        mock_client.cancel_order = AsyncMock(side_effect=Exception("order already filled"))
        mock_client.fetch_order = AsyncMock(return_value={
            "id": "order123",
            "status": "closed",
            "filled": 0.2,
            "average": 50005.0,
            "fee": {"cost": 2.0},
        })

        result = await executor._cancel_and_handle_remainder(state, reason="timeout")

        assert result is not None
        assert result.fill_price == 50005.0
        assert result.source == "SmartExecutor:limit_race"


class TestPartialFillAndMerge:
    """부분 체결 + VWAP merge 테스트."""

    async def test_partial_fill_remainder_market(self) -> None:
        """60% limit + 40% market → VWAP merge."""
        executor, mock_inner, mock_client = _make_smart_executor()

        # Poll: open, then after cancel: partial fill
        poll_count = 0

        async def mock_fetch_order(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal poll_count
            poll_count += 1
            if poll_count <= 1:
                return {"id": "order123", "status": "open", "filled": 0, "average": 0}
            # After cancel: partial fill (60% of 0.2 BTC)
            return {
                "id": "order123",
                "status": "canceled",
                "filled": 0.12,
                "average": 50005.0,
                "fee": {"cost": 1.2},
            }

        mock_client.fetch_order = AsyncMock(side_effect=mock_fetch_order)
        mock_client.fetch_ticker = AsyncMock(return_value={"bid": 50000.0, "ask": 50010.0})

        # Market fallback for remainder
        market_fill = FillEvent(
            client_order_id="test-mkt",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50020.0,
            fill_qty=0.08,
            fee=1.6,
            fill_timestamp=datetime.now(UTC),
            source="LiveExecutor",
        )
        mock_inner.execute = AsyncMock(return_value=market_fill)

        order = _make_order(notional_usd=10000.0)

        # Patch time so we exceed deadline on first poll
        with patch("asyncio.sleep", new_callable=AsyncMock), \
             patch("asyncio.get_event_loop") as mock_loop:
            mock_time = MagicMock()
            mock_time.time = MagicMock(side_effect=[0.0, 0.0, 100.0])
            mock_loop.return_value = mock_time
            result = await executor.execute(order)

        assert result is not None
        # VWAP: (0.12 * 50005 + 0.08 * 50020) / (0.12 + 0.08)
        expected_vwap = (0.12 * 50005.0 + 0.08 * 50020.0) / 0.20
        assert abs(result.fill_price - expected_vwap) < 0.01
        assert abs(result.fill_qty - 0.20) < 1e-10
        assert abs(result.fee - 2.8) < 0.01

    def test_merge_fills_vwap_correct(self) -> None:
        """VWAP 계산 정확성."""
        limit_fill = FillEvent(
            client_order_id="test-1",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50000.0,
            fill_qty=0.3,
            fee=3.0,
            fill_timestamp=datetime.now(UTC),
            source="SmartExecutor:limit_partial",
        )
        market_fill = FillEvent(
            client_order_id="test-1-mkt",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=50100.0,
            fill_qty=0.2,
            fee=4.0,
            fill_timestamp=datetime.now(UTC),
            source="LiveExecutor",
        )

        merged = SmartExecutor._merge_fills(limit_fill, market_fill, "test-1")

        # VWAP = (0.3 * 50000 + 0.2 * 50100) / 0.5 = 50040
        assert abs(merged.fill_price - 50040.0) < 0.01
        assert abs(merged.fill_qty - 0.5) < 1e-10
        assert abs(merged.fee - 7.0) < 0.01
        assert merged.client_order_id == "test-1"
        assert merged.source == "SmartExecutor:merged"


class TestStaleCleanup:
    """시작 시 미체결 정리 테스트."""

    async def test_stale_cleanup_on_startup(self) -> None:
        """시작 시 미체결 limit 주문 취소."""
        executor, _, mock_client = _make_smart_executor()

        mock_client.fetch_open_orders = AsyncMock(return_value=[
            {"id": "stale1", "type": "limit", "symbol": "BTC/USDT:USDT"},
            {"id": "stale2", "type": "market", "symbol": "BTC/USDT:USDT"},  # market은 무시
        ])

        await executor.cleanup_stale_orders(["BTC/USDT"])

        # limit만 취소
        mock_client.cancel_order.assert_called_once_with("stale1", "BTC/USDT:USDT")


class TestExecutorPortSatisfied:
    """ExecutorPort Protocol 만족 테스트."""

    def test_executor_port_satisfied(self) -> None:
        """SmartExecutor는 ExecutorPort Protocol을 만족."""
        from src.eda.ports import ExecutorPort

        executor, _, _ = _make_smart_executor()
        assert isinstance(executor, ExecutorPort)


class TestSetPmDelegation:
    """set_pm delegation 테스트."""

    def test_set_pm_delegates_to_inner(self) -> None:
        """set_pm은 inner에 위임."""
        executor, mock_inner, _ = _make_smart_executor()
        mock_pm = MagicMock()
        executor.set_pm(mock_pm)
        mock_inner.set_pm.assert_called_once_with(mock_pm)


class TestLimitPriceComputation:
    """Limit 가격 계산 테스트."""

    def test_buy_limit_below_ask(self) -> None:
        """BUY: ask보다 약간 낮은 가격."""
        executor, _, _ = _make_smart_executor()
        price = executor._compute_limit_price("BUY", bid=50000.0, ask=50010.0)
        assert price < 50010.0
        assert price > 50000.0  # ask * (1 - offset) 범위

    def test_sell_limit_above_bid(self) -> None:
        """SELL: bid보다 약간 높은 가격."""
        executor, _, _ = _make_smart_executor()
        price = executor._compute_limit_price("SELL", bid=50000.0, ask=50010.0)
        assert price > 50000.0
        assert price < 50010.0
