"""Executor 테스트.

BacktestExecutor의 next-open 체결, 수수료 계산을 검증합니다.
"""

from datetime import UTC, datetime
from uuid import uuid4

from src.core.events import BarEvent, OrderRequestEvent
from src.eda.executors import BacktestExecutor
from src.portfolio.cost_model import CostModel


def _make_bar(
    symbol: str = "BTC/USDT",
    open_price: float = 50000.0,
    close_price: float = 50500.0,
) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timeframe="1D",
        open=open_price,
        high=max(open_price, close_price) * 1.01,
        low=min(open_price, close_price) * 0.99,
        close=close_price,
        volume=1000.0,
        bar_timestamp=datetime.now(UTC),
    )


def _make_order(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    notional_usd: float = 10000.0,
) -> OrderRequestEvent:
    return OrderRequestEvent(
        client_order_id=f"test-{uuid4().hex[:8]}",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        target_weight=0.5,
        notional_usd=notional_usd,
        validated=True,
        correlation_id=uuid4(),
        source="test",
    )


class TestBacktestExecutor:
    """BacktestExecutor 테스트."""

    async def test_fill_at_open_price(self) -> None:
        """Open 가격으로 체결."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar(open_price=48000.0, close_price=50000.0))

        fill = await executor.execute(_make_order(notional_usd=9600.0))
        assert fill is not None
        assert fill.fill_price == 48000.0
        assert abs(fill.fill_qty - 0.2) < 1e-10  # 9600 / 48000

    async def test_fee_calculation(self) -> None:
        """CostModel 수수료 적용."""
        cost_model = CostModel.binance_futures()  # total_fee_rate ~0.11%
        executor = BacktestExecutor(cost_model=cost_model)
        executor.on_bar(_make_bar(open_price=50000.0))

        order = _make_order(notional_usd=10000.0)
        fill = await executor.execute(order)
        assert fill is not None

        expected_fee = 10000.0 * cost_model.total_fee_rate
        assert abs(fill.fee - expected_fee) < 0.01

    async def test_no_price_returns_none(self) -> None:
        """가격 데이터 없으면 None 반환."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        # on_bar 호출 안 함

        fill = await executor.execute(_make_order(symbol="UNKNOWN/USDT"))
        assert fill is None

    async def test_multi_symbol_prices(self) -> None:
        """멀티 심볼 독립적 가격 추적."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar(symbol="BTC/USDT", open_price=50000.0))
        executor.on_bar(_make_bar(symbol="ETH/USDT", open_price=3000.0))

        btc_fill = await executor.execute(_make_order(symbol="BTC/USDT", notional_usd=10000.0))
        eth_fill = await executor.execute(_make_order(symbol="ETH/USDT", notional_usd=3000.0))

        assert btc_fill is not None
        assert btc_fill.fill_price == 50000.0

        assert eth_fill is not None
        assert eth_fill.fill_price == 3000.0

    async def test_fill_preserves_correlation_id(self) -> None:
        """FillEvent가 주문의 correlation_id를 보존."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        order = _make_order()
        fill = await executor.execute(order)
        assert fill is not None
        assert fill.correlation_id == order.correlation_id

    async def test_fill_preserves_side(self) -> None:
        """SELL 주문은 SELL fill."""
        executor = BacktestExecutor(cost_model=CostModel.zero())
        executor.on_bar(_make_bar())

        fill = await executor.execute(_make_order(side="SELL"))
        assert fill is not None
        assert fill.side == "SELL"
