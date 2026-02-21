"""EventBus 핸들러 priority 테스트.

priority 기반 핸들러 실행 순서를 검증합니다.
"""

from __future__ import annotations

import pytest

from src.core.event_bus import EventBus
from src.core.events import BarEvent, EventType


def _make_bar() -> BarEvent:
    """테스트용 BarEvent 생성."""
    from datetime import UTC, datetime

    return BarEvent(
        symbol="BTC/USDT",
        timeframe="1D",
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=100.0,
        bar_timestamp=datetime(2025, 6, 1, tzinfo=UTC),
        source="test",
    )


class TestEventBusPriority:
    """EventBus priority 기반 핸들러 실행 순서 검증."""

    @pytest.mark.asyncio
    async def test_handlers_execute_in_priority_order(self) -> None:
        """낮은 priority 핸들러가 먼저 실행된다."""
        bus = EventBus(queue_size=100)
        execution_order: list[str] = []

        async def handler_analytics(event: object) -> None:
            execution_order.append("analytics")

        async def handler_strategy(event: object) -> None:
            execution_order.append("strategy")

        async def handler_regime(event: object) -> None:
            execution_order.append("regime")

        # 의도적으로 역순 등록
        bus.subscribe(EventType.BAR, handler_analytics, priority=200)
        bus.subscribe(EventType.BAR, handler_strategy, priority=50)
        bus.subscribe(EventType.BAR, handler_regime, priority=10)

        bar = _make_bar()
        await bus._dispatch(bar)

        assert execution_order == ["regime", "strategy", "analytics"]

    @pytest.mark.asyncio
    async def test_same_priority_preserves_insertion_order(self) -> None:
        """동일 priority 시 등록 순서(seq)를 보장한다."""
        bus = EventBus(queue_size=100)
        execution_order: list[str] = []

        async def handler_a(event: object) -> None:
            execution_order.append("a")

        async def handler_b(event: object) -> None:
            execution_order.append("b")

        async def handler_c(event: object) -> None:
            execution_order.append("c")

        # 모두 동일 priority — 등록 순서 보장
        bus.subscribe(EventType.BAR, handler_a, priority=100)
        bus.subscribe(EventType.BAR, handler_b, priority=100)
        bus.subscribe(EventType.BAR, handler_c, priority=100)

        bar = _make_bar()
        await bus._dispatch(bar)

        assert execution_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_default_priority_is_100(self) -> None:
        """priority 미지정 시 기본값 100."""
        bus = EventBus(queue_size=100)
        execution_order: list[str] = []

        async def handler_default(event: object) -> None:
            execution_order.append("default")

        async def handler_high_priority(event: object) -> None:
            execution_order.append("high")

        async def handler_low_priority(event: object) -> None:
            execution_order.append("low")

        # default (100) → high (10) → low (200)
        bus.subscribe(EventType.BAR, handler_default)  # priority=100 (기본)
        bus.subscribe(EventType.BAR, handler_high_priority, priority=10)
        bus.subscribe(EventType.BAR, handler_low_priority, priority=200)

        bar = _make_bar()
        await bus._dispatch(bar)

        assert execution_order == ["high", "default", "low"]

    @pytest.mark.asyncio
    async def test_eda_component_priority_order(self) -> None:
        """EDA 컴포넌트 권장 priority 순서: Regime→Feature→Strategy→PM→RM→OMS→Analytics."""
        bus = EventBus(queue_size=100)
        execution_order: list[str] = []

        async def regime_handler(event: object) -> None:
            execution_order.append("regime")

        async def feature_handler(event: object) -> None:
            execution_order.append("feature")

        async def strategy_handler(event: object) -> None:
            execution_order.append("strategy")

        async def pm_handler(event: object) -> None:
            execution_order.append("pm")

        async def rm_handler(event: object) -> None:
            execution_order.append("rm")

        async def oms_handler(event: object) -> None:
            execution_order.append("oms")

        async def analytics_handler(event: object) -> None:
            execution_order.append("analytics")

        # 의도적으로 무작위 순서로 등록 — priority가 순서를 결정
        bus.subscribe(EventType.BAR, analytics_handler, priority=200)
        bus.subscribe(EventType.BAR, pm_handler, priority=100)
        bus.subscribe(EventType.BAR, regime_handler, priority=10)
        bus.subscribe(EventType.BAR, oms_handler, priority=120)
        bus.subscribe(EventType.BAR, strategy_handler, priority=50)
        bus.subscribe(EventType.BAR, feature_handler, priority=20)
        bus.subscribe(EventType.BAR, rm_handler, priority=110)

        bar = _make_bar()
        await bus._dispatch(bar)

        assert execution_order == [
            "regime",
            "feature",
            "strategy",
            "pm",
            "rm",
            "oms",
            "analytics",
        ]

    @pytest.mark.asyncio
    async def test_mixed_event_types_independent(self) -> None:
        """서로 다른 EventType의 priority는 독립적이다."""
        bus = EventBus(queue_size=100)

        async def noop(event: object) -> None:
            pass

        bus.subscribe(EventType.BAR, noop, priority=50)
        bus.subscribe(EventType.FILL, noop, priority=10)

        # BAR 핸들러: priority=50
        assert bus._handlers[EventType.BAR][0][0] == 50
        # FILL 핸들러: priority=10
        assert bus._handlers[EventType.FILL][0][0] == 10

    def test_handler_tuple_structure(self) -> None:
        """핸들러가 (priority, seq, handler) 튜플로 저장된다."""
        bus = EventBus(queue_size=100)

        async def my_handler(event: object) -> None:
            pass

        bus.subscribe(EventType.BAR, my_handler, priority=42)

        handlers = bus._handlers[EventType.BAR]
        assert len(handlers) == 1
        priority, seq, handler = handlers[0]
        assert priority == 42
        assert isinstance(seq, int)
        assert handler is my_handler

    @pytest.mark.asyncio
    async def test_error_isolation_with_priority(self) -> None:
        """priority 핸들러에서 에러 발생해도 후속 핸들러가 실행된다."""
        bus = EventBus(queue_size=100)
        execution_order: list[str] = []

        async def failing_handler(event: object) -> None:
            execution_order.append("fail")
            msg = "intentional error"
            raise RuntimeError(msg)

        async def later_handler(event: object) -> None:
            execution_order.append("later")

        bus.subscribe(EventType.BAR, failing_handler, priority=10)
        bus.subscribe(EventType.BAR, later_handler, priority=20)

        bar = _make_bar()
        await bus._dispatch(bar)

        assert execution_order == ["fail", "later"]
        assert bus.metrics.handler_errors == 1

    @pytest.mark.asyncio
    async def test_seq_monotonic(self) -> None:
        """seq 값이 등록 순서대로 단조 증가한다."""
        bus = EventBus(queue_size=100)

        async def h1(event: object) -> None:
            pass

        async def h2(event: object) -> None:
            pass

        async def h3(event: object) -> None:
            pass

        bus.subscribe(EventType.BAR, h1, priority=100)
        bus.subscribe(EventType.FILL, h2, priority=100)
        bus.subscribe(EventType.BAR, h3, priority=100)

        # BAR 핸들러 seq 확인
        bar_handlers = bus._handlers[EventType.BAR]
        assert bar_handlers[0][1] < bar_handlers[1][1]  # h1.seq < h3.seq
