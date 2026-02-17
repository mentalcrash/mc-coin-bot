"""EDA PortfolioManager 테스트.

Signal → Order 변환, Fill → 포지션 업데이트, 잔고 추적을 검증합니다.
Equity 정확성, Position Stop-Loss, Trailing Stop, BalanceUpdate on Bar 검증.
"""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from src.core.event_bus import EventBus
from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    BarEvent,
    EventType,
    FillEvent,
    OrderRequestEvent,
    PositionUpdateEvent,
    SignalEvent,
)
from src.eda.portfolio_manager import EDAPortfolioManager
from src.models.types import Direction
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel


def _make_signal(
    symbol: str = "BTC/USDT",
    direction: Direction = Direction.LONG,
    strength: float = 1.0,
    bar_timestamp: datetime | None = None,
) -> SignalEvent:
    return SignalEvent(
        symbol=symbol,
        strategy_name="tsmom",
        direction=direction,
        strength=strength,
        bar_timestamp=bar_timestamp or datetime.now(UTC),
        correlation_id=uuid4(),
        source="test",
    )


def _make_fill(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    price: float = 50000.0,
    qty: float = 0.1,
    fee: float = 1.1,
) -> FillEvent:
    return FillEvent(
        client_order_id="test-1",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        fill_price=price,
        fill_qty=qty,
        fee=fee,
        fill_timestamp=datetime.now(UTC),
        correlation_id=uuid4(),
        source="test",
    )


class TestSignalToOrder:
    """SignalEvent -> OrderRequestEvent 변환."""

    async def test_signal_generates_order(self) -> None:
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_signal(direction=Direction.LONG, strength=1.5))
        await bus.stop()
        await task

        assert len(orders) == 1
        assert orders[0].symbol == "BTC/USDT"
        assert orders[0].side == "BUY"
        assert orders[0].target_weight > 0

    async def test_leverage_clamping(self) -> None:
        """strength > max_leverage_cap 시 클램핑."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # strength=5.0 → 2.0으로 클램핑
        await bus.publish(_make_signal(strength=5.0))
        await bus.stop()
        await task

        assert len(orders) == 1
        assert orders[0].target_weight <= 2.0

    async def test_rebalance_threshold_filter(self) -> None:
        """변화가 rebalance_threshold 미만이면 주문 생성 안 함."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.10)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # strength=0.05, weight = 0.05 → 0에서 0.05 변화 < 0.10 threshold
        await bus.publish(_make_signal(strength=0.05))
        await bus.stop()
        await task

        assert len(orders) == 0

    async def test_neutral_signal_target_weight_zero(self) -> None:
        """NEUTRAL 시그널은 target_weight=0."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_signal(direction=Direction.NEUTRAL, strength=0.0))
        await bus.stop()
        await task

        # 0에서 0으로 → 변화 없음 → 주문 없음
        assert len(orders) == 0

    async def test_asset_weights_applied(self) -> None:
        """멀티 에셋 가중치 적용 (배치 모드 flush 통해 검증)."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(
            config=config,
            initial_capital=10000.0,
            asset_weights={"BTC/USDT": 0.5, "ETH/USDT": 0.5},
        )
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # strength=1.0, weight=0.5 → target=0.5
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=t1))
        await bus.flush()

        # 배치 모드 → flush_pending_signals로 주문 생성
        await pm.flush_pending_signals()
        await bus.flush()
        await bus.stop()
        await task

        assert len(orders) == 1
        assert orders[0].target_weight == 0.5


class TestFillProcessing:
    """FillEvent → 포지션/잔고 업데이트."""

    async def test_fill_updates_position(self) -> None:
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        pos_events: list[PositionUpdateEvent] = []
        bal_events: list[BalanceUpdateEvent] = []

        async def pos_handler(event: AnyEvent) -> None:
            if isinstance(event, PositionUpdateEvent):
                pos_events.append(event)

        async def bal_handler(event: AnyEvent) -> None:
            if isinstance(event, BalanceUpdateEvent):
                bal_events.append(event)

        bus.subscribe(EventType.POSITION_UPDATE, pos_handler)
        bus.subscribe(EventType.BALANCE_UPDATE, bal_handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=2.2))
        await bus.stop()
        await task

        assert len(pos_events) == 1
        assert pos_events[0].direction == Direction.LONG
        assert pos_events[0].size == 0.1

        assert len(bal_events) == 1
        # cash = 10000 - (50000*0.1) - 2.2 = 4997.8
        assert abs(bal_events[0].available_cash - 4997.8) < 0.1

    async def test_close_long_position(self) -> None:
        """롱 포지션 청산 시 실현 손익 계산."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        pos_events: list[PositionUpdateEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, PositionUpdateEvent):
                pos_events.append(event)

        bus.subscribe(EventType.POSITION_UPDATE, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # 진입: BUY 0.1 @ 50000
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        # 청산: SELL 0.1 @ 51000 (이익: +100)
        await bus.publish(_make_fill(side="SELL", price=51000.0, qty=0.1, fee=0.0))
        await bus.stop()
        await task

        assert len(pos_events) == 2
        close_pos = pos_events[1]
        assert close_pos.direction == Direction.NEUTRAL
        assert close_pos.size == 0.0
        assert abs(close_pos.realized_pnl - 100.0) < 0.01

    async def test_unique_client_order_ids(self) -> None:
        """client_order_id 고유성."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_signal(strength=1.0))
        await bus.publish(_make_signal(strength=0.5))  # 변화 → 새 주문
        await bus.stop()
        await task

        if len(orders) >= 2:
            ids = [o.client_order_id for o in orders]
            assert len(set(ids)) == len(ids)  # 모두 고유


class TestPortfolioManagerProperties:
    """PM 속성 테스트."""

    def test_initial_state(self) -> None:
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        assert pm.total_equity == 10000.0
        assert pm.available_cash == 10000.0
        assert pm.aggregate_leverage == 0.0
        assert pm.open_position_count == 0


# =========================================================================
# Helper: BarEvent 생성
# =========================================================================
def _make_bar(
    symbol: str = "BTC/USDT",
    open_: float = 50000.0,
    high: float = 51000.0,
    low: float = 49000.0,
    close: float = 50500.0,
    volume: float = 100.0,
) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timeframe="1D",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_timestamp=datetime.now(UTC),
        correlation_id=uuid4(),
        source="test",
    )


# =========================================================================
# A. Equity 수정 검증
# =========================================================================
class TestEquityCalculation:
    """Equity 이중 계산 버그 수정 검증."""

    async def test_equity_no_double_count_long(self) -> None:
        """LONG 포지션: equity = cash + notional (unrealized 이중 계산 없음)."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # BUY 0.1 BTC @ 50000 → cash = 10000 - 5000 = 5000
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # 가격 상승 → 55000
        await bus.publish(_make_bar(close=55000.0, high=55000.0, low=50000.0))
        await bus.flush()
        await bus.stop()
        await task

        # notional = 0.1 * 55000 = 5500
        # equity = cash(5000) + notional(5500) = 10500
        assert abs(pm.total_equity - 10500.0) < 1.0
        # unrealized이 이중으로 더해지면 11500이 됨 (오류)
        assert pm.total_equity < 11000.0

    async def test_equity_no_double_count_short(self) -> None:
        """SHORT 포지션: equity = cash - notional."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # SELL 0.1 BTC @ 50000 → cash = 10000 + 5000 = 15000
        await bus.publish(_make_fill(side="SELL", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # 가격 하락 → 45000 (SHORT 이익)
        await bus.publish(_make_bar(close=45000.0, high=50000.0, low=45000.0))
        await bus.flush()
        await bus.stop()
        await task

        # notional = 0.1 * 45000 = 4500
        # equity = cash(15000) - notional(4500) = 10500
        assert abs(pm.total_equity - 10500.0) < 1.0

    async def test_equity_after_price_change(self) -> None:
        """가격 변동 후 equity 정확 추적."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # BUY 0.2 BTC @ 50000 → cash = 10000 - 10000 = 0
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.2, fee=0.0))
        await bus.flush()

        # 가격 하락 → 48000 (손실)
        await bus.publish(_make_bar(close=48000.0, high=50000.0, low=47000.0))
        await bus.flush()
        await bus.stop()
        await task

        # notional = 0.2 * 48000 = 9600
        # equity = 0 + 9600 = 9600 (= 10000 - 400 loss)
        assert abs(pm.total_equity - 9600.0) < 1.0


# =========================================================================
# B. Position Stop-Loss 검증
# =========================================================================
class TestPositionStopLoss:
    """Position 레벨 손절 테스트."""

    async def test_stop_loss_long_triggered(self) -> None:
        """LONG: low < entry * (1-sl) → 청산 주문 발행."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=0.10,  # 10% stop-loss
            use_intrabar_stop=True,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # BUY 0.1 @ 50000
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # low = 44000 < 50000 * 0.9 = 45000 → stop-loss 발동
        await bus.publish(
            _make_bar(
                close=44500.0,
                high=50000.0,
                low=44000.0,
            )
        )
        await bus.flush()
        await bus.stop()
        await task

        # 청산 주문이 발행되어야 함
        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) >= 1
        assert close_orders[0].side == "SELL"

    async def test_stop_loss_long_not_triggered(self) -> None:
        """LONG: low > entry * (1-sl) → 유지."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=0.10,
            use_intrabar_stop=True,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # low = 46000 > 45000 → stop-loss 미발동
        await bus.publish(
            _make_bar(
                close=47000.0,
                high=50000.0,
                low=46000.0,
            )
        )
        await bus.flush()
        await bus.stop()
        await task

        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) == 0

    async def test_stop_loss_short_triggered(self) -> None:
        """SHORT: high > entry * (1+sl) → 청산."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=0.10,
            use_intrabar_stop=True,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # SELL 0.1 @ 50000
        await bus.publish(_make_fill(side="SELL", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # high = 56000 > 50000 * 1.1 = 55000 → stop-loss 발동
        await bus.publish(
            _make_bar(
                close=55500.0,
                high=56000.0,
                low=50000.0,
            )
        )
        await bus.flush()
        await bus.stop()
        await task

        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) >= 1
        assert close_orders[0].side == "BUY"

    async def test_stop_loss_close_based(self) -> None:
        """use_intrabar_stop=False → Close 기준 손절."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=0.10,
            use_intrabar_stop=False,  # Close 기준
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # low=44000(intrabar 기준이면 발동) 이지만 close=46000 > 45000 → 미발동
        await bus.publish(
            _make_bar(
                close=46000.0,
                high=50000.0,
                low=44000.0,
            )
        )
        await bus.flush()
        await bus.stop()
        await task

        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) == 0

    async def test_stop_loss_disabled(self) -> None:
        """system_stop_loss=None → 손절 체크 안 함."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # 가격 대폭 하락해도 stop-loss 미작동
        await bus.publish(
            _make_bar(
                close=30000.0,
                high=50000.0,
                low=30000.0,
            )
        )
        await bus.flush()
        await bus.stop()
        await task

        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) == 0


# =========================================================================
# C. Trailing Stop 검증
# =========================================================================
class TestTrailingStop:
    """ATR 기반 Trailing Stop 테스트."""

    def _make_pm_with_trailing(
        self,
        multiplier: float = 2.0,
        stop_loss: float | None = None,
    ) -> EDAPortfolioManager:
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=stop_loss,
            use_trailing_stop=True,
            trailing_stop_atr_multiplier=multiplier,
            cost_model=CostModel.zero(),
        )
        return EDAPortfolioManager(config=config, initial_capital=10000.0)

    async def test_trailing_stop_long_triggered(self) -> None:
        """LONG: close < peak - atr*mult → 청산."""
        pm = self._make_pm_with_trailing(multiplier=2.0)
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # BUY 0.1 @ 50000
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # 부드러운 상승 16봉 (ATR warmup + peak)
        # 각 봉 range=1000, 인접 봉 close 차이 ~625 → TR ≈ 1000 유지
        for i in range(16):
            base = 50000.0 + i * 625
            await bus.publish(
                _make_bar(
                    close=base + 500,
                    high=base + 1000,
                    low=base,
                )
            )
            await bus.flush()

        pos = pm.positions["BTC/USDT"]
        peak = pos.peak_price_since_entry
        # peak ≈ 60000 + 1000 = 61000 (last bar high)

        # ATR ≈ 1000, mult=2.0 → trailing distance ≈ 2000
        # close = peak - 2500 < peak - 2000 → trailing stop 발동
        await bus.publish(
            _make_bar(
                close=peak - 2500,
                high=peak - 500,
                low=peak - 3000,
            )
        )
        await bus.flush()
        await bus.stop()
        await task

        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) >= 1

    async def test_trailing_stop_warmup_no_trigger(self) -> None:
        """ATR 14봉 미달 시 trailing stop 비활성."""
        pm = self._make_pm_with_trailing(multiplier=2.0)
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # 10봉만 (14 미달) → ATR None → trailing stop 미작동
        for _ in range(10):
            await bus.publish(
                _make_bar(
                    close=40000.0,
                    high=50000.0,
                    low=40000.0,
                )
            )
            await bus.flush()

        await bus.stop()
        await task

        # warmup 미달이므로 trailing stop 청산 주문 없음
        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) == 0

    async def test_trailing_stop_disabled(self) -> None:
        """use_trailing_stop=False → 체크 안 함."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            use_trailing_stop=False,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # 충분한 봉 + 큰 하락 → trailing stop 비활성이면 청산 없음
        for i in range(20):
            await bus.publish(
                _make_bar(
                    close=50000.0 - i * 500,
                    high=50000.0,
                    low=50000.0 - i * 1000,
                )
            )
            await bus.flush()

        await bus.stop()
        await task

        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) == 0

    async def test_peak_tracks_high(self) -> None:
        """Peak이 bar.high를 따라 갱신."""
        pm = self._make_pm_with_trailing(multiplier=2.0)
        bus = EventBus(queue_size=100)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # 고점 상승 시리즈
        await bus.publish(_make_bar(close=51000.0, high=52000.0, low=50000.0))
        await bus.flush()
        await bus.publish(_make_bar(close=53000.0, high=55000.0, low=52000.0))
        await bus.flush()
        await bus.publish(_make_bar(close=54000.0, high=56000.0, low=53000.0))
        await bus.flush()

        await bus.stop()
        await task

        pos = pm.positions["BTC/USDT"]
        assert pos.peak_price_since_entry == 56000.0


# =========================================================================
# D. Balance Update on Bar 검증
# =========================================================================
class TestBalanceUpdateOnBar:
    """매 bar마다 BalanceUpdateEvent 발행 검증."""

    async def test_balance_update_emitted_on_bar(self) -> None:
        """포지션이 있으면 매 bar마다 BalanceUpdateEvent 발행."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        bal_events: list[BalanceUpdateEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, BalanceUpdateEvent):
                bal_events.append(event)

        bus.subscribe(EventType.BALANCE_UPDATE, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        fill_bal_count = len(bal_events)  # fill에서 발행된 것

        # 3개 bar → 3개 추가 BalanceUpdateEvent
        for _ in range(3):
            await bus.publish(_make_bar(close=51000.0, high=52000.0, low=50000.0))
            await bus.flush()

        await bus.stop()
        await task

        # fill 1개 + bar 3개 = 총 4개
        assert len(bal_events) == fill_bal_count + 3

    async def test_balance_update_not_emitted_no_position(self) -> None:
        """포지션 없으면 bar에서 BalanceUpdateEvent 미발행."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        bal_events: list[BalanceUpdateEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, BalanceUpdateEvent):
                bal_events.append(event)

        bus.subscribe(EventType.BALANCE_UPDATE, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # 포지션 없이 bar 발행
        await bus.publish(_make_bar(close=50000.0, high=51000.0, low=49000.0))
        await bus.flush()
        await bus.stop()
        await task

        assert len(bal_events) == 0


# =========================================================================
# E. Position 필드 초기화 검증
# =========================================================================
class TestPositionFieldInit:
    """진입 시 peak/trough 초기화 검증."""

    async def test_long_entry_initializes_peak(self) -> None:
        """LONG 진입 시 peak = entry price."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()
        await bus.stop()
        await task

        pos = pm.positions["BTC/USDT"]
        assert pos.peak_price_since_entry == 50000.0
        assert pos.trough_price_since_entry == 0.0

    async def test_short_entry_initializes_trough(self) -> None:
        """SHORT 진입 시 trough = entry price."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="SELL", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()
        await bus.stop()
        await task

        pos = pm.positions["BTC/USDT"]
        assert pos.trough_price_since_entry == 50000.0
        assert pos.peak_price_since_entry == 0.0

    async def test_close_resets_peak_trough(self) -> None:
        """포지션 청산 시 peak/trough 리셋."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()
        await bus.publish(_make_fill(side="SELL", price=51000.0, qty=0.1, fee=0.0))
        await bus.flush()
        await bus.stop()
        await task

        pos = pm.positions["BTC/USDT"]
        assert pos.peak_price_since_entry == 0.0
        assert pos.trough_price_since_entry == 0.0


# =========================================================================
# F. Vol-Target Rebalancing 검증
# =========================================================================
class TestVolTargetRebalancing:
    """Per-bar vol-target 리밸런싱 테스트."""

    async def test_rebalance_on_signal_change(self) -> None:
        """SignalEvent strength 변화 → PM이 리밸런스 주문 생성."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.05,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # 첫 시그널: strength=1.0 (target_weight=1.0)
        await bus.publish(_make_signal(strength=1.0))
        await bus.flush()
        assert len(orders) == 1

        # Fill 처리
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.2, fee=0.0))
        await bus.flush()

        # 두 번째 시그널: strength=0.5 (target_weight=0.5)
        # current_weight ≈ 1.0, target=0.5, diff=0.5 > 0.05 threshold
        await bus.publish(_make_signal(strength=0.5))
        await bus.flush()

        await bus.stop()
        await task

        # 최소 2개 주문 (진입 + 리밸런스)
        assert len(orders) >= 2

    async def test_rebalance_on_market_drift(self) -> None:
        """같은 signal이지만 가격 변동으로 current_weight drift → _on_bar에서 리밸런스."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.05,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # 시그널: LONG strength=1.0 → target_weight=1.0
        await bus.publish(_make_signal(strength=1.0))
        await bus.flush()
        initial_order_count = len(orders)
        assert initial_order_count == 1

        # Fill: 0.2 BTC @ 50000 → notional=10000, equity=10000, weight=1.0
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.2, fee=0.0))
        await bus.flush()

        # weight drift는 multi-asset에서 더 뚜렷함
        # 단일 에셋은 target=1.0이면 항상 weight=1.0이 유지
        # target을 0.5로 설정해서 drift 테스트
        pm._last_target_weights["BTC/USDT"] = 0.5

        # 가격 변동 bar → _on_bar에서 rebalance 체크
        await bus.publish(_make_bar(close=50000.0, high=51000.0, low=49000.0))
        await bus.flush()

        await bus.stop()
        await task

        # per-bar rebalancing이 주문을 생성해야 함
        # current_weight≈1.0, target=0.5, diff=0.5 > 0.05
        assert len(orders) > initial_order_count

    async def test_no_rebalance_below_threshold(self) -> None:
        """weight diff < rebalance_threshold → 주문 미생성."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.10,  # 10% threshold
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # 시그널: LONG strength=1.0
        await bus.publish(_make_signal(strength=1.0))
        await bus.flush()

        # Fill
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.2, fee=0.0))
        await bus.flush()
        order_count_after_entry = len(orders)

        # 동일 시그널 반복 → target_weight 동일 → diff ≈ 0 < 0.10
        await bus.publish(_make_signal(strength=1.0))
        await bus.flush()

        await bus.stop()
        await task

        # 추가 주문 없음
        assert len(orders) == order_count_after_entry

    async def test_stop_loss_prevents_reentry(self) -> None:
        """stop-loss 발동 → 같은 bar에서 시그널 수신해도 재진입 안 함."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=0.10,  # 10% stop-loss
            use_intrabar_stop=True,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # 시그널 + 진입
        await bus.publish(_make_signal(strength=1.0))
        await bus.flush()
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        orders_before_sl = len(orders)

        # stop-loss 발동 bar (low=44000 < 45000)
        sl_bar = _make_bar(close=44500.0, high=50000.0, low=44000.0)
        await bus.publish(sl_bar)
        await bus.flush()

        # stop-loss 주문이 발생
        sl_orders = [o for o in orders[orders_before_sl:] if o.target_weight == 0.0]
        assert len(sl_orders) >= 1

        order_count_after_sl = len(orders)

        # 같은 bar에서 새 시그널 → 재진입 차단
        # _stopped_this_bar에 BTC/USDT가 있으므로 _evaluate_rebalance가 return
        await bus.publish(_make_signal(strength=1.0))
        await bus.flush()

        await bus.stop()
        await task

        # stop-loss 후 추가 주문 없음
        assert len(orders) == order_count_after_sl

    async def test_multiple_bars_multiple_rebalances(self) -> None:
        """10+ bars 시뮬레이션 → 여러 리밸런스 주문 생성 확인."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.05,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        bus = EventBus(queue_size=500)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # 진입 시그널 + fill
        await bus.publish(_make_signal(strength=1.5))
        await bus.flush()
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.3, fee=0.0))
        await bus.flush()

        # 10 bars에서 strength를 교대로 변화 → 리밸런스 발생
        for i in range(10):
            strength = 1.0 if i % 2 == 0 else 1.5
            await bus.publish(_make_signal(strength=strength))
            await bus.flush()

        await bus.stop()
        await task

        # 최초 진입 + 여러 리밸런스 → 2개 이상 주문
        assert len(orders) >= 2


# =========================================================================
# G. Intrabar SL/TS 테스트 (target_timeframe 모드)
# =========================================================================
def _make_bar_tf(
    symbol: str = "BTC/USDT",
    timeframe: str = "1D",
    open_: float = 50000.0,
    high: float = 51000.0,
    low: float = 49000.0,
    close: float = 50500.0,
    volume: float = 100.0,
    ts: datetime | None = None,
) -> BarEvent:
    """타임프레임 지정 가능한 BarEvent 생성."""
    return BarEvent(
        symbol=symbol,
        timeframe=timeframe,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_timestamp=ts or datetime.now(UTC),
        correlation_id=uuid4(),
        source="test",
    )


class TestIntrabarStopLoss:
    """1m bar에서의 intrabar SL/TS 동작 검증."""

    async def test_1m_bar_triggers_stop_loss(self) -> None:
        """target_timeframe='1D' → 1m bar에서 SL 발동 → 청산 주문."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=0.10,
            use_intrabar_stop=True,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0, target_timeframe="1D")
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # BUY 0.1 @ 50000
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # 1m bar: low=44000 < 50000 * 0.9 = 45000 → SL 발동
        await bus.publish(
            _make_bar_tf(
                timeframe="1m",
                close=44500.0,
                high=50000.0,
                low=44000.0,
            )
        )
        await bus.flush()
        await bus.stop()
        await task

        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) >= 1
        assert close_orders[0].side == "SELL"

    async def test_1m_bar_no_rebalancing(self) -> None:
        """1m bar에서는 rebalancing/BalanceUpdate 생략."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0, target_timeframe="1D")
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []
        bal_events: list[BalanceUpdateEvent] = []

        async def order_handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        async def bal_handler(event: AnyEvent) -> None:
            if isinstance(event, BalanceUpdateEvent):
                bal_events.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, order_handler)
        bus.subscribe(EventType.BALANCE_UPDATE, bal_handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        # BUY + target weight 설정
        await bus.publish(_make_signal(strength=1.0))
        await bus.flush()
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.2, fee=0.0))
        await bus.flush()

        orders_before = len(orders)
        bal_before = len(bal_events)

        # 1m bar 여러 개 → rebalancing/BalanceUpdate 발생 안 함
        for i in range(5):
            await bus.publish(
                _make_bar_tf(
                    timeframe="1m",
                    close=50000.0 + i * 100,
                    high=50200.0 + i * 100,
                    low=49800.0 + i * 100,
                )
            )
            await bus.flush()

        await bus.stop()
        await task

        # 1m bar에서는 rebalance 주문 없음
        assert len(orders) == orders_before
        # 1m bar에서는 BalanceUpdate 없음
        assert len(bal_events) == bal_before

    async def test_tf_bar_full_logic(self) -> None:
        """target_timeframe='1D' → 1D bar에서 기존 전체 로직 동작."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0, target_timeframe="1D")
        bus = EventBus(queue_size=200)
        bal_events: list[BalanceUpdateEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, BalanceUpdateEvent):
                bal_events.append(event)

        bus.subscribe(EventType.BALANCE_UPDATE, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()
        bal_before = len(bal_events)

        # 1D bar → 전체 로직 (BalanceUpdate 발행)
        await bus.publish(
            _make_bar_tf(
                timeframe="1D",
                close=51000.0,
                high=52000.0,
                low=50000.0,
            )
        )
        await bus.flush()
        await bus.stop()
        await task

        assert len(bal_events) > bal_before

    async def test_tf_bar_triggers_full_logic(self) -> None:
        """target_timeframe='1D' → 1D bar에서 전체 로직 실행."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0, target_timeframe="1D")
        bus = EventBus(queue_size=200)
        bal_events: list[BalanceUpdateEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, BalanceUpdateEvent):
                bal_events.append(event)

        bus.subscribe(EventType.BALANCE_UPDATE, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()
        bal_before = len(bal_events)

        # 1D bar → 전체 로직 실행
        await bus.publish(_make_bar(close=51000.0, high=52000.0, low=50000.0))
        await bus.flush()
        await bus.stop()
        await task

        assert len(bal_events) > bal_before

    async def test_1m_trailing_stop_triggered(self) -> None:
        """1m bar에서 trailing stop 발동."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            use_trailing_stop=True,
            trailing_stop_atr_multiplier=2.0,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0, target_timeframe="1D")
        bus = EventBus(queue_size=500)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0))
        await bus.flush()

        # 1m bar로 ATR warmup (16봉, 부드러운 상승)
        for i in range(16):
            base_price = 50000.0 + i * 625
            await bus.publish(
                _make_bar_tf(
                    timeframe="1m",
                    close=base_price + 500,
                    high=base_price + 1000,
                    low=base_price,
                )
            )
            await bus.flush()

        pos = pm.positions["BTC/USDT"]
        peak = pos.peak_price_since_entry
        # ATR ≈ 1000, mult=2.0 → trailing distance ≈ 2000

        # 1m bar: close < peak - 2000 → trailing stop 발동
        await bus.publish(
            _make_bar_tf(
                timeframe="1m",
                close=peak - 2500,
                high=peak - 500,
                low=peak - 3000,
            )
        )
        await bus.flush()
        await bus.stop()
        await task

        close_orders = [o for o in orders if o.target_weight == 0.0]
        assert len(close_orders) >= 1


# =========================================================================
# H. Batch Order Processing (멀티에셋)
# =========================================================================
class TestBatchOrderProcessing:
    """멀티에셋 배치 주문 처리 테스트."""

    def _make_multi_pm(
        self,
        rebalance_threshold: float = 0.10,
        stop_loss: float | None = None,
    ) -> EDAPortfolioManager:
        """8-asset EW PM 생성 (batch_mode=True)."""
        symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "BNB/USDT",
            "ADA/USDT",
            "XRP/USDT",
            "DOT/USDT",
            "AVAX/USDT",
        ]
        weights = dict.fromkeys(symbols, 0.125)
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=rebalance_threshold,
            system_stop_loss=stop_loss,
            use_intrabar_stop=True,
            cost_model=CostModel.zero(),
        )
        return EDAPortfolioManager(
            config=config,
            initial_capital=10000.0,
            asset_weights=weights,
        )

    async def test_batch_mode_activated(self) -> None:
        """멀티에셋 → _batch_mode=True."""
        pm = self._make_multi_pm()
        assert pm._batch_mode is True

    async def test_single_asset_no_batch(self) -> None:
        """단일에셋 → _batch_mode=False."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        assert pm._batch_mode is False

    async def test_signals_collected_not_executed(self) -> None:
        """배치 모드: signal 수신 시 즉시 주문 생성하지 않음."""
        pm = self._make_multi_pm()
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # 동일 timestamp의 signal들 → 수집만 됨
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=ts))
        await bus.flush()
        await bus.publish(_make_signal(symbol="ETH/USDT", strength=1.0, bar_timestamp=ts))
        await bus.flush()

        await bus.stop()
        await task

        # 같은 timestamp → flush 안 됨 → 주문 없음
        assert len(orders) == 0
        assert len(pm._pending_signals) == 2

    async def test_flush_on_new_timestamp(self) -> None:
        """새 timestamp signal → 이전 배치 flush."""
        pm = self._make_multi_pm(rebalance_threshold=0.10)
        bus = EventBus(queue_size=500)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # T1: 8개 signal 수집
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        symbols = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "BNB/USDT",
            "ADA/USDT",
            "XRP/USDT",
            "DOT/USDT",
            "AVAX/USDT",
        ]
        for sym in symbols:
            await bus.publish(_make_signal(symbol=sym, strength=1.0, bar_timestamp=t1))
            await bus.flush()

        assert len(orders) == 0  # 아직 flush 안 됨

        # T2: 새 timestamp → T1 배치 flush
        t2 = datetime(2024, 1, 2, tzinfo=UTC)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=t2))
        await bus.flush()

        await bus.stop()
        await task

        # T1의 8개 signal이 flush → 8개 주문
        # (VBT parity: last_executed=0, target=0.125≠0 → special case → 모두 실행)
        assert len(orders) == 8

    async def test_same_equity_for_all_orders(self) -> None:
        """배치 flush 시 모든 주문이 동일 equity 기반."""
        pm = self._make_multi_pm(rebalance_threshold=0.10)
        bus = EventBus(queue_size=500)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # T1: 2개 signal
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=t1))
        await bus.flush()
        await bus.publish(_make_signal(symbol="ETH/USDT", strength=1.0, bar_timestamp=t1))
        await bus.flush()

        # T2: flush 트리거
        t2 = datetime(2024, 1, 2, tzinfo=UTC)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=t2))
        await bus.flush()

        await bus.stop()
        await task

        # 2개 주문 생성
        assert len(orders) == 2
        # 동일 equity 기반 → notional 비율 동일 (둘 다 weight=0.125)
        assert abs(orders[0].notional_usd - orders[1].notional_usd) < 1.0

    async def test_threshold_vbt_parity(self) -> None:
        """VBT parity: |new_target - last_executed| < threshold → 주문 미생성.

        첫 진입 후 동일 target 반복 시 rebalancing 안 함.
        """
        pm = self._make_multi_pm(rebalance_threshold=0.10)
        bus = EventBus(queue_size=500)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # T1: strength=1.0 → target=0.125, last_executed=0 → 첫 진입 (special case)
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=t1))
        await bus.flush()

        # T2: flush T1 (1개 주문) + 수집 T2
        t2 = datetime(2024, 1, 2, tzinfo=UTC)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=t2))
        await bus.flush()
        assert len(orders) == 1  # T1 flush: 첫 진입

        # T3: flush T2. target=0.125, last_executed=0.125, change=0 < 0.10 → skip
        t3 = datetime(2024, 1, 3, tzinfo=UTC)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=t3))
        await bus.flush()

        await bus.stop()
        await task

        # T2 flush에서 추가 주문 없음 (동일 target)
        assert len(orders) == 1

    async def test_flush_pending_signals_shutdown(self) -> None:
        """Runner shutdown 시 마지막 batch flush."""
        pm = self._make_multi_pm(rebalance_threshold=0.10)
        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # signal 수집 (flush 트리거 없이 종료)
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=t1))
        await bus.flush()
        await bus.publish(_make_signal(symbol="ETH/USDT", strength=1.0, bar_timestamp=t1))
        await bus.flush()

        assert len(orders) == 0

        # 명시적 flush (Runner가 호출)
        await pm.flush_pending_signals()
        await bus.flush()

        await bus.stop()
        await task

        assert len(orders) == 2

    async def test_stop_loss_prevents_batch_execution(self) -> None:
        """stop-loss 발동된 symbol은 batch flush 시 스킵."""
        pm = self._make_multi_pm(rebalance_threshold=0.10, stop_loss=0.10)
        bus = EventBus(queue_size=500)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # BTC 포지션 생성
        await bus.publish(
            _make_fill(symbol="BTC/USDT", side="BUY", price=50000.0, qty=0.025, fee=0.0)
        )
        await bus.flush()

        # BTC stop-loss 발동 bar
        ts = datetime(2024, 1, 2, tzinfo=UTC)
        await bus.publish(
            _make_bar_tf(
                symbol="BTC/USDT",
                timeframe="1D",
                close=44000.0,
                high=50000.0,
                low=43000.0,
                ts=ts,
            )
        )
        await bus.flush()

        # BTC가 _stopped_this_bar에 있음
        assert "BTC/USDT" in pm._stopped_this_bar

        sl_orders = len(orders)

        # signal 수집 (BTC, ETH 동일 timestamp)
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0, bar_timestamp=ts))
        await bus.flush()
        await bus.publish(_make_signal(symbol="ETH/USDT", strength=1.0, bar_timestamp=ts))
        await bus.flush()

        # flush 트리거 (Runner가 하는 것처럼 직접 flush)
        await pm.flush_pending_signals()
        await bus.flush()

        await bus.stop()
        await task

        # ETH만 주문 생성 (BTC는 SL bar에서 signal 수집 안 됨)
        batch_orders = orders[sl_orders:]
        assert len(batch_orders) == 1
        assert batch_orders[0].symbol == "ETH/USDT"

    async def test_batch_no_bar_drift_rebalancing(self) -> None:
        """배치 모드: _on_bar에서 per-bar drift rebalancing 안 함."""
        pm = self._make_multi_pm(rebalance_threshold=0.10)
        bus = EventBus(queue_size=500)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # BTC 포지션 생성 + target weight 설정
        await bus.publish(
            _make_fill(symbol="BTC/USDT", side="BUY", price=50000.0, qty=0.025, fee=0.0)
        )
        await bus.flush()
        pm._last_target_weights["BTC/USDT"] = 0.5  # 강제 drift

        orders_before = len(orders)

        # bar 발행 → 배치 모드이므로 per-bar rebalancing 안 함
        await bus.publish(_make_bar(close=50000.0, high=51000.0, low=49000.0))
        await bus.flush()

        await bus.stop()
        await task

        assert len(orders) == orders_before

    async def test_single_asset_backward_compat(self) -> None:
        """단일에셋: 기존 즉시 실행 동작 유지."""
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        assert pm._batch_mode is False

        bus = EventBus(queue_size=200)
        orders: list[OrderRequestEvent] = []

        async def handler(event: AnyEvent) -> None:
            if isinstance(event, OrderRequestEvent):
                orders.append(event)

        bus.subscribe(EventType.ORDER_REQUEST, handler)
        await pm.register(bus)

        task = asyncio.create_task(bus.start())

        # signal → 즉시 주문 (배치 아님)
        await bus.publish(_make_signal(strength=1.0))
        await bus.flush()

        await bus.stop()
        await task

        assert len(orders) == 1


class TestCashSufficiencyGuard:
    """PM cash 음수 방지 가드 검증."""

    async def test_sell_order_always_allowed(self) -> None:
        """SELL(청산) 주문은 현금 부족에도 허용."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)

        # _has_sufficient_cash 직접 테스트
        pm._cash = -100.0
        assert pm._has_sufficient_cash(5000.0, "SELL") is True

    async def test_buy_allowed_when_sufficient_cash(self) -> None:
        """충분한 현금이 있으면 BUY 허용."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)

        assert pm._has_sufficient_cash(5000.0, "BUY") is True

    async def test_buy_blocked_when_exceeds_leverage_floor(self) -> None:
        """레버리지 한도를 초과하는 BUY 차단."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)

        # max_leverage_cap=2.0 → cash_floor = -(10000*(2-1) + 10) = -10010
        # cash=10000, notional=25000 → projected = -15000 < -10010 → 차단
        assert pm._has_sufficient_cash(25000.0, "BUY") is False

    async def test_buy_within_leverage_allowed(self) -> None:
        """레버리지 범위 내 BUY 허용 (notional > cash but within leverage)."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)

        # cash=10000, notional=15000 → projected = -5000 > -10010 → 허용
        assert pm._has_sufficient_cash(15000.0, "BUY") is True

    async def test_cash_guard_blocks_extreme_order(self) -> None:
        """극단적 notional이 레버리지 한도 초과 시 차단하는 통합 테스트."""
        config = PortfolioManagerConfig(max_leverage_cap=1.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)

        # max_leverage_cap=1.0 → cash_floor = -(10000*(1-1) + 10) = -10
        # notional 직접 테스트: cash=10000, notional=10020 → projected = -20 < -10 → 차단
        assert pm._has_sufficient_cash(10020.0, "BUY") is False
        # notional=10005 → projected = -5 > -10 → 허용
        assert pm._has_sufficient_cash(10005.0, "BUY") is True


class TestReconcileWithExchange:
    """reconcile_with_exchange() 테스트 — phantom position 제거."""

    def _make_pm_with_positions(
        self,
        positions: dict[str, tuple[float, Direction, float]],
    ) -> EDAPortfolioManager:
        """포지션이 있는 PM 생성. {symbol: (size, direction, last_price)}."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        for symbol, (size, direction, price) in positions.items():
            from src.eda.portfolio_manager import Position

            pm._positions[symbol] = Position(
                symbol=symbol,
                direction=direction,
                size=size,
                avg_entry_price=price,
                last_price=price,
                peak_price_since_entry=price,
                trough_price_since_entry=price,
            )
        return pm

    def test_remove_phantom_position(self) -> None:
        """거래소에 없는 PM 포지션(phantom) 제거."""
        pm = self._make_pm_with_positions({"BTC/USDT": (0.01, Direction.LONG, 50000.0)})
        # 거래소에 빈 포지션
        exchange: dict[str, tuple[float, Direction]] = {}
        removed = pm.reconcile_with_exchange(exchange)

        assert removed == ["BTC/USDT"]
        pos = pm.positions["BTC/USDT"]
        assert pos.size == 0.0
        assert pos.direction == Direction.NEUTRAL
        assert pos.avg_entry_price == 0.0

    def test_keep_matching_position(self) -> None:
        """거래소에도 있는 포지션은 유지."""
        pm = self._make_pm_with_positions({"BTC/USDT": (0.01, Direction.LONG, 50000.0)})
        exchange = {"BTC/USDT": (0.01, Direction.LONG)}
        removed = pm.reconcile_with_exchange(exchange)

        assert removed == []
        assert pm.positions["BTC/USDT"].size == 0.01

    def test_exchange_only_not_added(self) -> None:
        """거래소에만 있는 포지션은 PM에 추가하지 않음."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        exchange = {"ETH/USDT": (1.0, Direction.LONG)}
        removed = pm.reconcile_with_exchange(exchange)

        assert removed == []
        assert "ETH/USDT" not in pm.positions

    def test_auxiliary_state_cleanup(self) -> None:
        """phantom 제거 시 부가 상태도 정리."""
        pm = self._make_pm_with_positions({"BTC/USDT": (0.01, Direction.LONG, 50000.0)})
        # 부가 상태 설정
        pm._last_target_weights["BTC/USDT"] = 0.5
        pm._last_executed_targets["BTC/USDT"] = 0.5
        pm._pending_signals["BTC/USDT"] = 0.5
        pm._pending_close.add("BTC/USDT")
        pm._deferred_close_targets["BTC/USDT"] = 0.0

        removed = pm.reconcile_with_exchange({})

        assert removed == ["BTC/USDT"]
        assert "BTC/USDT" not in pm._last_target_weights
        assert "BTC/USDT" not in pm._last_executed_targets
        assert "BTC/USDT" not in pm._pending_signals
        assert "BTC/USDT" not in pm._pending_close
        assert "BTC/USDT" not in pm._deferred_close_targets

    def test_partial_removal(self) -> None:
        """여러 포지션 중 phantom만 제거."""
        pm = self._make_pm_with_positions(
            {
                "BTC/USDT": (0.01, Direction.LONG, 50000.0),
                "ETH/USDT": (1.0, Direction.SHORT, 3000.0),
            }
        )
        exchange = {"BTC/USDT": (0.01, Direction.LONG)}
        removed = pm.reconcile_with_exchange(exchange)

        assert removed == ["ETH/USDT"]
        assert pm.positions["BTC/USDT"].size == 0.01
        assert pm.positions["ETH/USDT"].size == 0.0

    def test_remove_all_phantoms(self) -> None:
        """모든 PM 포지션이 phantom일 때 전체 제거."""
        pm = self._make_pm_with_positions(
            {
                "BTC/USDT": (0.01, Direction.LONG, 50000.0),
                "ETH/USDT": (1.0, Direction.SHORT, 3000.0),
            }
        )
        removed = pm.reconcile_with_exchange({})

        assert set(removed) == {"BTC/USDT", "ETH/USDT"}
        assert all(not p.is_open for p in pm.positions.values())

    def test_both_empty(self) -> None:
        """PM과 거래소 모두 빈 포지션이면 빈 리스트."""
        config = PortfolioManagerConfig(max_leverage_cap=2.0, rebalance_threshold=0.01)
        pm = EDAPortfolioManager(config=config, initial_capital=10000.0)
        removed = pm.reconcile_with_exchange({})

        assert removed == []
