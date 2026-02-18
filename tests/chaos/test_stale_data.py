"""Chaos Test -- Stale data 시나리오.

WS gap, 중복 bar, 오래된 데이터 처리를 검증합니다.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from src.core.event_bus import EventBus
from src.core.events import BarEvent, EventType

pytestmark = pytest.mark.chaos


class TestStaleData:
    """Stale data 시나리오 테스트."""

    async def test_duplicate_bar_processing(self) -> None:
        """동일 timestamp의 중복 bar가 전달되어도 오류 없음."""
        bus = EventBus(queue_size=100)
        received: list[BarEvent] = []

        async def on_bar(event: object) -> None:
            assert isinstance(event, BarEvent)
            received.append(event)

        bus.subscribe(EventType.BAR, on_bar)  # type: ignore[arg-type]

        # 동일 timestamp의 bar 2개 생성
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000,
            high=51000,
            low=49000,
            close=50500,
            volume=1000,
            bar_timestamp=ts,
        )

        # 큐에 2번 publish
        await bus.publish(bar)
        await bus.publish(bar)

        # bus start → dispatch → check
        bus_task = await _start_and_flush(bus)
        assert len(received) == 2  # 중복 처리해도 에러 없음
        await _stop_bus(bus, bus_task)

    async def test_out_of_order_bars(self) -> None:
        """시간 역순 bar가 와도 처리 가능."""
        bus = EventBus(queue_size=100)
        received: list[BarEvent] = []

        async def on_bar(event: object) -> None:
            assert isinstance(event, BarEvent)
            received.append(event)

        bus.subscribe(EventType.BAR, on_bar)  # type: ignore[arg-type]

        now = datetime(2025, 1, 10, tzinfo=UTC)

        # 역순으로 발행
        for days_ago in [2, 0, 1]:
            ts = now - timedelta(days=days_ago)
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000,
                high=51000,
                low=49000,
                close=50500,
                volume=1000,
                bar_timestamp=ts,
            )
            await bus.publish(bar)

        bus_task = await _start_and_flush(bus)
        assert len(received) == 3
        await _stop_bus(bus, bus_task)

    async def test_gap_in_bars(self) -> None:
        """Bar 사이에 gap이 있어도 처리."""
        bus = EventBus(queue_size=100)
        received: list[BarEvent] = []

        async def on_bar(event: object) -> None:
            assert isinstance(event, BarEvent)
            received.append(event)

        bus.subscribe(EventType.BAR, on_bar)  # type: ignore[arg-type]

        # 1일, 5일, 10일 — gap 존재
        for day in [1, 5, 10]:
            ts = datetime(2025, 1, day, tzinfo=UTC)
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000 + day * 100,
                high=51000 + day * 100,
                low=49000 + day * 100,
                close=50500 + day * 100,
                volume=1000,
                bar_timestamp=ts,
            )
            await bus.publish(bar)

        bus_task = await _start_and_flush(bus)
        assert len(received) == 3
        # 가격이 bar_timestamp 순서와 무관하게 처리됨
        assert all(r.close > 0 for r in received)
        await _stop_bus(bus, bus_task)

    async def test_zero_volume_bar(self) -> None:
        """volume=0인 bar도 정상 처리."""
        bus = EventBus(queue_size=100)
        received: list[BarEvent] = []

        async def on_bar(event: object) -> None:
            assert isinstance(event, BarEvent)
            received.append(event)

        bus.subscribe(EventType.BAR, on_bar)  # type: ignore[arg-type]

        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=50000,
            high=51000,
            low=49000,
            close=50500,
            volume=0,
            bar_timestamp=datetime.now(UTC),
        )
        await bus.publish(bar)

        bus_task = await _start_and_flush(bus)
        assert len(received) == 1
        assert received[0].volume == 0
        await _stop_bus(bus, bus_task)


# ── Helpers ───────────────────────────────────────────────────────


async def _start_and_flush(bus: EventBus) -> asyncio.Task[None]:
    """bus를 시작하고 flush."""
    task = asyncio.create_task(bus.start())
    await bus.flush()
    return task


async def _stop_bus(bus: EventBus, task: asyncio.Task[None]) -> None:
    """bus를 정지."""
    await bus.stop()
    await task
