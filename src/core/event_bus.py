"""비동기 EventBus 구현.

In-process async EventBus로 bounded queue, backpressure, JSONL audit log를 지원합니다.
Phase 4-6은 단일 프로세스로 운용하며, 추후 Redis Streams 등으로 교체 가능합니다.

Rules Applied:
    - #10 Python Standards: asyncio.Queue, TaskGroup
    - #23 Exception Handling: handler 에러 격리
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.core.events import DROPPABLE_EVENTS, AnyEvent, EventType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


type EventHandler = Callable[[AnyEvent], Awaitable[None]]

# Backpressure: 드롭 가능 이벤트의 publish timeout (초)
_DROPPABLE_PUBLISH_TIMEOUT = 0.01

# 연속 drop 경고 임계값
_DROP_ALERT_THRESHOLD = 10


class EventBusMetrics:
    """EventBus 런타임 메트릭."""

    __slots__ = (
        "_consecutive_drops",
        "events_dispatched",
        "events_dropped",
        "events_published",
        "handler_errors",
        "max_queue_depth",
    )

    def __init__(self) -> None:
        self.events_published: int = 0
        self.events_dispatched: int = 0
        self.events_dropped: int = 0
        self.handler_errors: int = 0
        self.max_queue_depth: int = 0
        self._consecutive_drops: int = 0

    def record_drop(self) -> None:
        """이벤트 드롭 기록 + 연속 드롭 임계값 경고."""
        self.events_dropped += 1
        self._consecutive_drops += 1
        if self._consecutive_drops >= _DROP_ALERT_THRESHOLD:
            logger.critical(
                "ALERT: {} consecutive events dropped (total dropped: {}). Queue may be permanently congested.",
                self._consecutive_drops,
                self.events_dropped,
            )

    def record_publish(self) -> None:
        """이벤트 발행 성공 기록 + 연속 드롭 카운터 리셋."""
        self.events_published += 1
        self._consecutive_drops = 0

    def snapshot(self) -> dict[str, int]:
        """현재 메트릭 스냅샷."""
        return {
            "events_published": self.events_published,
            "events_dispatched": self.events_dispatched,
            "events_dropped": self.events_dropped,
            "handler_errors": self.handler_errors,
            "max_queue_depth": self.max_queue_depth,
        }


class EventBus:
    """In-process async EventBus.

    Type-safe subscribe, bounded queue backpressure, JSONL audit log를 지원합니다.

    사용법:
        bus = EventBus(queue_size=1000)
        bus.subscribe(EventType.BAR, on_bar_handler)
        await bus.start()  # 이벤트 소비 루프 시작
        await bus.publish(bar_event)  # 이벤트 발행
        await bus.stop()   # 큐 드레인 후 정지
    """

    def __init__(
        self,
        queue_size: int = 10000,
        event_log_path: str | None = None,
    ) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._queue: asyncio.Queue[AnyEvent | None] = asyncio.Queue(maxsize=queue_size)
        self._running = False
        self._event_log_path = event_log_path
        self._log_file: Any = None
        self._consumer_task: asyncio.Task[None] | None = None
        self.metrics = EventBusMetrics()

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """이벤트 타입에 핸들러를 등록합니다.

        Args:
            event_type: 구독할 이벤트 타입
            handler: async 핸들러 함수
        """
        self._handlers[event_type].append(handler)

    async def publish(self, event: AnyEvent) -> None:
        """이벤트를 큐에 발행합니다.

        Backpressure 정책:
        - DROPPABLE 이벤트 (BAR, HEARTBEAT): 큐 가득 시 드롭
        - NEVER_DROP 이벤트 (SIGNAL, FILL 등): 큐 여유 생길 때까지 대기

        Args:
            event: 발행할 이벤트
        """
        event_type = event.event_type

        if event_type in DROPPABLE_EVENTS:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                self.metrics.record_drop()
                logger.warning(
                    "Event dropped (queue full): type={} id={} (total dropped: {})",
                    event_type,
                    event.event_id,
                    self.metrics.events_dropped,
                )
                return
        else:
            await self._queue.put(event)

        self.metrics.record_publish()

        # 최대 큐 깊이 추적
        self.metrics.max_queue_depth = max(self.metrics.max_queue_depth, self._queue.qsize())

    async def start(self) -> None:
        """이벤트 소비 루프를 시작합니다.

        큐에서 이벤트를 꺼내어 등록된 핸들러에 dispatch합니다.
        None sentinel 수신 시 종료합니다.
        """
        if self._event_log_path:
            Path(self._event_log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = Path(self._event_log_path).open("a")  # noqa: ASYNC230, SIM115

        self._running = True
        logger.debug("EventBus started")

        while self._running:
            event = await self._queue.get()
            if event is None:
                # Sentinel: 종료 신호
                self._queue.task_done()
                break

            await self._dispatch(event)
            self._queue.task_done()

        # 종료 전 남은 이벤트 드레인
        await self._drain()

        if self._log_file:
            self._log_file.close()
            self._log_file = None

        logger.debug(
            "EventBus stopped: {}",
            self.metrics.snapshot(),
        )

    async def flush(self) -> None:
        """큐의 모든 이벤트가 처리될 때까지 대기합니다.

        bar-by-bar 백테스트에서 각 bar의 이벤트 체인
        (Bar → Signal → Order → Fill)이 완료되도록 보장합니다.
        """
        await self._queue.join()

    async def stop(self) -> None:
        """EventBus를 정지합니다.

        None sentinel을 큐에 넣어 소비 루프를 종료합니다.
        """
        self._running = False
        await self._queue.put(None)

    async def _dispatch(self, event: AnyEvent) -> None:
        """등록된 핸들러에 이벤트를 dispatch합니다.

        핸들러 에러는 로깅 후 계속 진행합니다 (에러 격리).

        Args:
            event: dispatch할 이벤트
        """
        event_type = event.event_type
        handlers = self._handlers.get(event_type, [])

        # JSONL 로그 기록
        if self._log_file:
            self._log_file.write(event.model_dump_json() + "\n")
            self._log_file.flush()

        for handler in handlers:
            try:
                await handler(event)
            except Exception as exc:
                self.metrics.handler_errors += 1
                logger.exception(
                    "Handler error: handler={} event_type={} event_id={}",
                    handler.__name__,
                    event_type,
                    event.event_id,
                )
                # Prometheus errors_counter (lazy import — core → monitoring 의존 방지)
                try:
                    from src.monitoring.metrics import errors_counter

                    errors_counter.labels(
                        component="EventBus", error_type=type(exc).__name__
                    ).inc()
                except Exception:  # noqa: S110
                    pass

        self.metrics.events_dispatched += 1

    async def _drain(self) -> None:
        """큐에 남아있는 이벤트를 모두 처리합니다."""
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                if event is not None:
                    await self._dispatch(event)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    @property
    def is_running(self) -> bool:
        """EventBus 실행 상태."""
        return self._running

    @property
    def queue_size(self) -> int:
        """현재 큐에 대기 중인 이벤트 수."""
        return self._queue.qsize()
