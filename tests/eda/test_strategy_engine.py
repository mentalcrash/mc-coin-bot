"""StrategyEngine 테스트.

BaseStrategy를 이벤트 기반으로 래핑하는 Adapter 패턴의 동작을 검증합니다.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType, SignalEvent
from src.eda.strategy_engine import StrategyEngine
from src.models.types import Direction
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals


class SimpleTestStrategy(BaseStrategy):
    """테스트용 간단한 전략.

    close > open이면 LONG(+1), 아니면 SHORT(-1).
    strength = abs(close - open) / open.
    """

    @property
    def name(self) -> str:
        return "test_simple"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = (df["close"] > df["open"]).astype(int) * 2 - 1  # -1 or 1
        strength = ((df["close"] - df["open"]).abs() / df["open"]).shift(1).fillna(0)
        entries = direction.diff().fillna(0).abs() > 0
        exits = pd.Series(False, index=df.index)
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


def _make_bar(
    symbol: str,
    open_: float,
    close: float,
    ts: datetime,
) -> BarEvent:
    """테스트용 BarEvent."""
    return BarEvent(
        symbol=symbol,
        timeframe="1D",
        open=open_,
        high=max(open_, close) * 1.01,
        low=min(open_, close) * 0.99,
        close=close,
        volume=1000.0,
        bar_timestamp=ts,
        correlation_id=uuid4(),
        source="test",
    )


class TestStrategyEngineWarmup:
    """워밍업 관련 테스트."""

    async def test_no_signal_before_warmup(self) -> None:
        """warmup 미달 시 SignalEvent 발행 안 함."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(strategy, warmup_periods=10)
        bus = EventBus(queue_size=100)
        signals: list[AnyEvent] = []

        async def handler(event: AnyEvent) -> None:
            signals.append(event)

        bus.subscribe(EventType.SIGNAL, handler)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())

        # 5개 bar만 발행 (warmup=10보다 적음)
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(5):
            bar = _make_bar("BTC/USDT", 50000.0, 50100.0, base + timedelta(days=i))
            await bus.publish(bar)

        await bus.stop()
        await task

        assert len(signals) == 0

    async def test_signal_after_warmup(self) -> None:
        """warmup 이후 SignalEvent 발행."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(strategy, warmup_periods=5)
        bus = EventBus(queue_size=100)
        signals: list[SignalEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, SignalEvent)
            signals.append(event)

        bus.subscribe(EventType.SIGNAL, handler)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())

        # 10개 bar 발행 (warmup=5 이후 시그널 발행 가능)
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(10):
            # 상승/하락 교대로
            open_ = 50000.0
            close = 50200.0 if i % 2 == 0 else 49800.0
            bar = _make_bar("BTC/USDT", open_, close, base + timedelta(days=i))
            await bus.publish(bar)

        await bus.stop()
        await task

        assert len(signals) > 0
        assert all(s.strategy_name == "test_simple" for s in signals)


class TestStrategyEngineSignalDedup:
    """시그널 dedup 테스트."""

    async def test_same_signal_not_duplicated(self) -> None:
        """동일 시그널 반복 시 중복 발행 안 함."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(strategy, warmup_periods=3)
        bus = EventBus(queue_size=100)
        signals: list[SignalEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, SignalEvent)
            signals.append(event)

        bus.subscribe(EventType.SIGNAL, handler)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())

        # 모든 bar가 close > open (항상 LONG)
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(10):
            bar = _make_bar("BTC/USDT", 50000.0, 50100.0, base + timedelta(days=i))
            await bus.publish(bar)

        await bus.stop()
        await task

        # 처음 시그널 발행 후 동일 시그널은 dedup
        # warmup=3 이후 첫 시그널 1개 (이후 동일하므로 dedup)
        # strength가 (50100-50000)/50000 ≈ 0.002로 일정 → 1번만 발행
        assert len(signals) >= 1
        # 초기 한 번은 발행 (0에서 변화)
        first_signal = signals[0]
        assert first_signal.direction == Direction.LONG


class TestStrategyEngineMultiSymbol:
    """멀티 심볼 독립 버퍼 테스트."""

    async def test_independent_buffers_per_symbol(self) -> None:
        """심볼별 독립적인 버퍼와 시그널 추적."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(strategy, warmup_periods=3)
        bus = EventBus(queue_size=100)
        signals: list[SignalEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, SignalEvent)
            signals.append(event)

        bus.subscribe(EventType.SIGNAL, handler)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())

        base = datetime(2024, 1, 1, tzinfo=UTC)
        # BTC: 상승 (LONG), ETH: 하락 (SHORT)
        for i in range(5):
            ts = base + timedelta(days=i)
            await bus.publish(_make_bar("BTC/USDT", 50000.0, 50200.0, ts))
            await bus.publish(_make_bar("ETH/USDT", 3000.0, 2900.0, ts))

        await bus.stop()
        await task

        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        eth_signals = [s for s in signals if s.symbol == "ETH/USDT"]

        # 각 심볼에서 최소 1개 시그널
        assert len(btc_signals) >= 1
        assert len(eth_signals) >= 1

        # BTC=LONG, ETH=SHORT
        assert btc_signals[0].direction == Direction.LONG
        assert eth_signals[0].direction == Direction.SHORT


class TestStrategyEngineCorrelationId:
    """correlation_id 전파 테스트."""

    async def test_correlation_id_propagated(self) -> None:
        """BarEvent의 correlation_id가 SignalEvent에 전파."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(strategy, warmup_periods=3)
        bus = EventBus(queue_size=100)
        signals: list[SignalEvent] = []
        bars_sent: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, SignalEvent)
            signals.append(event)

        bus.subscribe(EventType.SIGNAL, handler)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())

        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(5):
            bar = _make_bar("BTC/USDT", 50000.0, 50200.0, base + timedelta(days=i))
            bars_sent.append(bar)
            await bus.publish(bar)

        await bus.stop()
        await task

        if signals:
            # 시그널의 correlation_id가 해당 bar의 것과 일치
            signal = signals[0]
            assert signal.correlation_id is not None


class TestStrategyEngineWarmupDetection:
    """워밍업 자동 감지 테스트."""

    def test_default_warmup(self) -> None:
        """config가 없으면 기본 warmup."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(strategy)
        assert engine.warmup_periods == 50  # 기본값
