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
) -> SignalEvent:
    return SignalEvent(
        symbol=symbol,
        strategy_name="tsmom",
        direction=direction,
        strength=strength,
        bar_timestamp=datetime.now(UTC),
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
        """멀티 에셋 가중치 적용."""
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
        await bus.publish(_make_signal(symbol="BTC/USDT", strength=1.0))
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
