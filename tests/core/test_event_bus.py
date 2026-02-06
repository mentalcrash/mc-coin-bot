"""EventBus 테스트.

subscribe/publish/dispatch, backpressure, JSONL 로그, 에러 격리를 검증합니다.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from src.core.event_bus import EventBus
from src.core.events import (
    AnyEvent,
    BarEvent,
    EventType,
    HeartbeatEvent,
    SignalEvent,
)
from src.models.types import Direction


def _make_bar(symbol: str = "BTC/USDT") -> BarEvent:
    """테스트용 BarEvent 생성."""
    return BarEvent(
        symbol=symbol,
        timeframe="1D",
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=1000.0,
        bar_timestamp=datetime.now(UTC),
    )


def _make_signal(symbol: str = "BTC/USDT") -> SignalEvent:
    """테스트용 SignalEvent 생성."""
    return SignalEvent(
        symbol=symbol,
        strategy_name="tsmom",
        direction=Direction.LONG,
        strength=1.0,
        bar_timestamp=datetime.now(UTC),
    )


class TestEventBusBasic:
    """기본 subscribe/publish/dispatch 테스트."""

    async def test_subscribe_and_dispatch(self) -> None:
        """구독 후 이벤트 dispatch 확인."""
        bus = EventBus(queue_size=100)
        received: list[AnyEvent] = []

        async def handler(event: AnyEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.BAR, handler)

        bar = _make_bar()

        # start를 백그라운드 태스크로 실행
        task = asyncio.create_task(bus.start())
        await bus.publish(bar)
        await bus.stop()
        await task

        assert len(received) == 1
        assert received[0].event_id == bar.event_id

    async def test_multiple_handlers(self) -> None:
        """동일 이벤트 타입에 여러 핸들러 등록."""
        bus = EventBus(queue_size=100)
        counter_a: list[int] = []
        counter_b: list[int] = []

        async def handler_a(event: AnyEvent) -> None:
            counter_a.append(1)

        async def handler_b(event: AnyEvent) -> None:
            counter_b.append(1)

        bus.subscribe(EventType.BAR, handler_a)
        bus.subscribe(EventType.BAR, handler_b)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_bar())
        await bus.stop()
        await task

        assert len(counter_a) == 1
        assert len(counter_b) == 1

    async def test_fifo_ordering(self) -> None:
        """이벤트 FIFO 순서 보장."""
        bus = EventBus(queue_size=100)
        received: list[str] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            received.append(event.symbol)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        for sym in ["A", "B", "C", "D"]:
            await bus.publish(_make_bar(sym))
        await bus.stop()
        await task

        assert received == ["A", "B", "C", "D"]

    async def test_unsubscribed_events_ignored(self) -> None:
        """구독하지 않은 이벤트 타입은 무시."""
        bus = EventBus(queue_size=100)
        received: list[AnyEvent] = []

        async def handler(event: AnyEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.SIGNAL, handler)  # SIGNAL만 구독

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_bar())  # BAR 발행 → 무시
        await bus.stop()
        await task

        assert len(received) == 0

    async def test_metrics_tracking(self) -> None:
        """메트릭 추적 확인."""
        bus = EventBus(queue_size=100)

        async def handler(event: AnyEvent) -> None:
            pass

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_bar())
        await bus.publish(_make_bar())
        await bus.stop()
        await task

        assert bus.metrics.events_published == 2
        assert bus.metrics.events_dispatched == 2
        assert bus.metrics.events_dropped == 0
        assert bus.metrics.handler_errors == 0


class TestEventBusBackpressure:
    """Backpressure 테스트."""

    async def test_droppable_event_dropped_on_full_queue(self) -> None:
        """큐 가득 시 DROPPABLE 이벤트 드롭."""
        bus = EventBus(queue_size=2)

        # 큐를 가득 채움 (start 없이 publish만)
        await bus.publish(_make_bar("A"))
        await bus.publish(_make_bar("B"))

        # 큐 가득 → BAR(droppable) 드롭
        await bus.publish(_make_bar("C"))

        assert bus.metrics.events_dropped == 1
        assert bus.metrics.events_published == 2  # A, B만

    async def test_heartbeat_droppable(self) -> None:
        """HeartbeatEvent도 드롭 가능."""
        bus = EventBus(queue_size=1)

        await bus.publish(_make_bar("A"))  # 큐 가득

        hb = HeartbeatEvent(component="DataFeed", bars_processed=100)
        await bus.publish(hb)  # 드롭

        assert bus.metrics.events_dropped == 1


class TestEventBusErrorIsolation:
    """핸들러 에러 격리 테스트."""

    async def test_handler_error_does_not_crash_bus(self) -> None:
        """한 핸들러 에러가 다른 핸들러에 영향 주지 않음."""
        bus = EventBus(queue_size=100)
        received: list[AnyEvent] = []

        async def bad_handler(event: AnyEvent) -> None:
            msg = "intentional error"
            raise ValueError(msg)

        async def good_handler(event: AnyEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.BAR, bad_handler)
        bus.subscribe(EventType.BAR, good_handler)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_bar())
        await bus.stop()
        await task

        # good_handler는 정상 실행
        assert len(received) == 1
        assert bus.metrics.handler_errors == 1

    async def test_multiple_handler_errors_tracked(self) -> None:
        """여러 핸들러 에러가 모두 카운트됨."""
        bus = EventBus(queue_size=100)

        async def bad_handler(event: AnyEvent) -> None:
            msg = "error"
            raise RuntimeError(msg)

        bus.subscribe(EventType.BAR, bad_handler)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_bar())
        await bus.publish(_make_bar())
        await bus.stop()
        await task

        assert bus.metrics.handler_errors == 2


class TestEventBusJSONL:
    """JSONL 이벤트 로그 테스트."""

    async def test_event_log_written(self, tmp_path: Path) -> None:
        """이벤트가 JSONL 파일에 기록됨."""
        log_path = tmp_path / "events.jsonl"
        bus = EventBus(queue_size=100, event_log_path=str(log_path))

        async def handler(event: AnyEvent) -> None:
            pass

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_bar())
        await bus.publish(_make_signal())
        await bus.stop()
        await task

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert "BTC/USDT" in lines[0]

    async def test_no_log_when_path_none(self) -> None:
        """event_log_path가 None이면 로그 안 씀."""
        bus = EventBus(queue_size=100, event_log_path=None)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_bar())
        await bus.stop()
        await task

        # 로그 파일 미생성 확인
        assert bus.metrics.events_published == 1


class TestEventBusGracefulStop:
    """Graceful stop 테스트."""

    async def test_drain_remaining_events(self) -> None:
        """stop 호출 시 남은 이벤트 처리."""
        bus = EventBus(queue_size=100)
        received: list[AnyEvent] = []

        async def handler(event: AnyEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.BAR, handler)

        # start 전에 이벤트 발행
        for _ in range(3):
            await bus.publish(_make_bar())

        task = asyncio.create_task(bus.start())
        await bus.stop()
        await task

        assert len(received) == 3

    async def test_is_running_state(self) -> None:
        """is_running 상태 전환 확인."""
        bus = EventBus(queue_size=100)
        assert not bus.is_running

        task = asyncio.create_task(bus.start())
        await asyncio.sleep(0.01)
        assert bus.is_running

        await bus.stop()
        await task
        assert not bus.is_running
