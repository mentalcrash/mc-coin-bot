"""Executor 테스트.

BacktestExecutor의 deferred execution, SL/TS 즉시 체결, 수수료 계산을 검증합니다.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from src.core.events import BarEvent, OrderRequestEvent
from src.eda.executors import BacktestExecutor
from src.portfolio.cost_model import CostModel


def _make_bar(
    symbol: str = "BTC/USDT",
    open_price: float = 50000.0,
    close_price: float = 50500.0,
    ts: datetime | None = None,
    timeframe: str = "1D",
) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timeframe=timeframe,
        open=open_price,
        high=max(open_price, close_price) * 1.01,
        low=min(open_price, close_price) * 0.99,
        close=close_price,
        volume=1000.0,
        bar_timestamp=ts or datetime.now(UTC),
    )


def _make_order(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    notional_usd: float = 10000.0,
    price: float | None = None,
) -> OrderRequestEvent:
    return OrderRequestEvent(
        client_order_id=f"test-{uuid4().hex[:8]}",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        target_weight=0.5,
        notional_usd=notional_usd,
        price=price,
        validated=True,
        correlation_id=uuid4(),
        source="test",
    )


class TestDeferredExecution:
    """Deferred execution (일반 주문 → 다음 bar open 체결)."""

    async def test_normal_order_deferred(self) -> None:
        """일반 주문(price=None)은 즉시 체결되지 않고 pending."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar(open_price=48000.0))

        fill = await executor.execute(_make_order())
        assert fill is None  # deferred
        assert executor.pending_count == 1

    async def test_deferred_fill_at_next_bar_open(self) -> None:
        """다음 bar 도착 시 open 가격으로 체결."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        bar1 = _make_bar(open_price=48000.0, close_price=49000.0)
        executor.on_bar(bar1)

        await executor.execute(_make_order(notional_usd=9600.0))

        # 다음 bar 도착 → fill
        bar2 = _make_bar(open_price=50000.0, close_price=51000.0)
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        assert len(fills) == 1
        assert fills[0].fill_price == 50000.0  # bar2 open
        assert abs(fills[0].fill_qty - 0.192) < 1e-10  # 9600 / 50000

    async def test_deferred_fill_timestamp(self) -> None:
        """Deferred fill의 timestamp는 체결 bar의 timestamp."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        ts1 = datetime(2024, 1, 1, tzinfo=UTC)
        ts2 = datetime(2024, 1, 2, tzinfo=UTC)

        executor.on_bar(_make_bar(ts=ts1))
        await executor.execute(_make_order())

        executor.on_bar(_make_bar(ts=ts2))
        executor.fill_pending(_make_bar(ts=ts2))

        fills = executor.drain_fills()
        assert fills[0].fill_timestamp == ts2  # bar2 timestamp

    async def test_pending_count_decreases_after_fill(self) -> None:
        """fill_pending 후 pending_count 감소."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        await executor.execute(_make_order())
        await executor.execute(_make_order())
        assert executor.pending_count == 2

        executor.on_bar(_make_bar())
        executor.fill_pending(_make_bar())

        assert executor.pending_count == 0
        fills = executor.drain_fills()
        assert len(fills) == 2

    async def test_drain_fills_clears_buffer(self) -> None:
        """drain_fills() 호출 후 비워짐."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())
        await executor.execute(_make_order())

        bar2 = _make_bar()
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills1 = executor.drain_fills()
        fills2 = executor.drain_fills()
        assert len(fills1) == 1
        assert len(fills2) == 0  # 이미 비워짐

    async def test_multi_symbol_deferred_fills(self) -> None:
        """멀티 심볼: 해당 심볼 bar 도착 시에만 fill."""
        executor = BacktestExecutor(cost_model=CostModel.zero())

        # BTC, ETH 각각 주문
        executor.on_bar(_make_bar(symbol="BTC/USDT", open_price=50000.0))
        executor.on_bar(_make_bar(symbol="ETH/USDT", open_price=3000.0))

        await executor.execute(_make_order(symbol="BTC/USDT", notional_usd=10000.0))
        await executor.execute(_make_order(symbol="ETH/USDT", notional_usd=3000.0))
        assert executor.pending_count == 2

        # BTC bar만 도착 → BTC만 fill
        btc_bar2 = _make_bar(symbol="BTC/USDT", open_price=51000.0)
        executor.on_bar(btc_bar2)
        executor.fill_pending(btc_bar2)

        assert executor.pending_count == 1  # ETH 아직 pending
        fills = executor.drain_fills()
        assert len(fills) == 1
        assert fills[0].symbol == "BTC/USDT"
        assert fills[0].fill_price == 51000.0

    async def test_deferred_preserves_correlation_id(self) -> None:
        """Deferred fill이 주문의 correlation_id를 보존."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        order = _make_order()
        await executor.execute(order)

        bar2 = _make_bar()
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        assert fills[0].correlation_id == order.correlation_id

    async def test_deferred_preserves_side(self) -> None:
        """SELL 주문은 SELL fill."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        await executor.execute(_make_order(side="SELL"))

        bar2 = _make_bar()
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        assert fills[0].side == "SELL"

    async def test_no_next_bar_stays_pending(self) -> None:
        """다음 bar 없이 종료 시 pending 유지."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        await executor.execute(_make_order())
        assert executor.pending_count == 1
        # drain_fills 호출해도 체결 안 됨
        assert len(executor.drain_fills()) == 0


class TestImmediateExecution:
    """SL/TS 즉시 체결 (order.price 설정)."""

    async def test_sl_order_fills_immediately(self) -> None:
        """SL 주문(price 설정)은 즉시 체결."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        fill = await executor.execute(_make_order(price=47000.0, notional_usd=9400.0))
        assert fill is not None
        assert fill.fill_price == 47000.0
        assert executor.pending_count == 0

    async def test_ts_order_fills_immediately(self) -> None:
        """TS 주문(price 설정)은 즉시 체결."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        bar_ts = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        executor.on_bar(_make_bar(ts=bar_ts))

        fill = await executor.execute(_make_order(price=52000.0, notional_usd=5200.0))
        assert fill is not None
        assert fill.fill_price == 52000.0
        assert fill.fill_timestamp == bar_ts

    async def test_mixed_immediate_and_deferred(self) -> None:
        """SL 즉시 + 일반 deferred 혼합."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        # SL → 즉시
        sl_fill = await executor.execute(_make_order(price=47000.0))
        assert sl_fill is not None

        # 일반 → deferred
        normal_fill = await executor.execute(_make_order())
        assert normal_fill is None
        assert executor.pending_count == 1


class TestFeeCalculation:
    """CostModel 수수료 적용 테스트."""

    async def test_fee_on_immediate_fill(self) -> None:
        """SL 즉시 체결 시 수수료 적용."""
        cost_model = CostModel.binance_futures()
        executor = BacktestExecutor(cost_model=cost_model)
        executor.on_bar(_make_bar())

        fill = await executor.execute(_make_order(price=50000.0, notional_usd=10000.0))
        assert fill is not None
        expected_fee = 10000.0 * cost_model.total_fee_rate
        assert abs(fill.fee - expected_fee) < 0.01

    async def test_fee_on_deferred_fill(self) -> None:
        """Deferred 체결 시 수수료 적용."""
        cost_model = CostModel.binance_futures()
        executor = BacktestExecutor(cost_model=cost_model)
        executor.on_bar(_make_bar())

        await executor.execute(_make_order(notional_usd=10000.0))

        bar2 = _make_bar(open_price=51000.0)
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        expected_fee = 10000.0 * cost_model.total_fee_rate
        assert abs(fills[0].fee - expected_fee) < 0.01


class TestEdgeCases:
    """엣지 케이스 테스트."""

    async def test_fill_at_zero_price_fails(self) -> None:
        """가격 0인 bar로 fill 시 실패."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())
        await executor.execute(_make_order())

        bar2 = _make_bar(open_price=0.0)
        executor.on_bar(bar2)
        executor.fill_pending(bar2)

        fills = executor.drain_fills()
        assert len(fills) == 0  # invalid price → skip
        assert executor.pending_count == 0  # order consumed even if fill fails

    async def test_fill_timestamp_updates_per_bar(self) -> None:
        """각 bar마다 fill_timestamp 업데이트."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        ts1 = datetime(2024, 1, 1, tzinfo=UTC)
        ts2 = ts1 + timedelta(days=1)
        ts3 = ts2 + timedelta(days=1)

        # 첫 주문
        executor.on_bar(_make_bar(ts=ts1))
        await executor.execute(_make_order())

        # 두 번째 bar에서 fill
        bar2 = _make_bar(ts=ts2)
        executor.on_bar(bar2)
        executor.fill_pending(bar2)
        fills1 = executor.drain_fills()
        assert fills1[0].fill_timestamp == ts2

        # 다시 주문 + 세 번째 bar에서 fill
        await executor.execute(_make_order())
        bar3 = _make_bar(ts=ts3)
        executor.on_bar(bar3)
        executor.fill_pending(bar3)
        fills2 = executor.drain_fills()
        assert fills2[0].fill_timestamp == ts3
