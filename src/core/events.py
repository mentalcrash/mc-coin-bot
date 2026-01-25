"""Event-Driven Architecture core components."""

import asyncio
from collections.abc import Callable, Coroutine
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event types in the system."""

    # Data events
    MARKET_DATA = "MARKET_DATA"
    CANDLE_CLOSED = "CANDLE_CLOSED"

    # Strategy events
    SIGNAL = "SIGNAL"

    # Order events
    ORDER_REQUEST = "ORDER_REQUEST"
    ORDER_APPROVED = "ORDER_APPROVED"
    ORDER_REJECTED = "ORDER_REJECTED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"

    # System events
    HEARTBEAT = "HEARTBEAT"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


class Event(BaseModel):
    """Base event class."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Any = None
    source: str = "system"  # Component that emitted the event


# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Central event bus for EDA.

    All components communicate through this bus.
    Supports async handlers and multiple subscribers per event type.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._running = False
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._background_tasks: set[asyncio.Task[None]] = set()

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Handler subscribed to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)
            logger.debug(f"Handler unsubscribed from {event_type.value}")

    async def emit(self, event: Event) -> None:
        """Emit an event to all subscribers."""
        await self._queue.put(event)

    def emit_sync(self, event: Event) -> None:
        """Emit an event synchronously (for non-async contexts)."""
        task = asyncio.create_task(self.emit(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _process_event(self, event: Event) -> None:
        """Process a single event by calling all handlers."""
        handlers = self._handlers.get(event.type, [])

        if not handlers:
            logger.trace(f"No handlers for event type: {event.type.value}")
            return

        logger.debug(
            f"Processing event {event.type.value} (id={event.id[:8]}) with {len(handlers)} handlers"
        )

        # Run all handlers concurrently
        tasks = [handler(event) for handler in handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any errors
        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    f"Handler error for {event.type.value}: {result}",
                    exc_info=result,
                )

    async def start(self) -> None:
        """Start the event processing loop."""
        self._running = True
        logger.info("EventBus started")

        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._process_event(event)
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"EventBus error: {e}", exc_info=True)

    async def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False
        # Process remaining events
        while not self._queue.empty():
            event = await self._queue.get()
            await self._process_event(event)
        logger.info("EventBus stopped")

    @property
    def is_running(self) -> bool:
        """Check if event bus is running."""
        return self._running
