"""Bounded Queue Sink for backpressure handling.

This module solves the memory safety issue with loguru's `enqueue=True`
which uses an unbounded SimpleQueue. With slow sinks (network, disk I/O),
the unbounded queue causes indefinite memory growth and eventual OOM.

See: https://github.com/Delgan/loguru/issues/1419

Rules Applied:
    - #15 Logging Standards: Non-blocking I/O for trading bots
    - #10 Python Standards: Modern typing, threading best practices
"""

from __future__ import annotations

import atexit
import queue
import sys
import threading
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class DropPolicy(Enum):
    """Policy for handling queue overflow.

    Attributes:
        OLDEST: Drop the oldest message in queue to make room (default)
        NEWEST: Drop the incoming message (silent discard)
        BLOCK: Block the caller until space is available (not recommended for HFT)
    """

    OLDEST = "oldest"
    NEWEST = "newest"
    BLOCK = "block"


class BoundedQueueSink:
    """Backpressure-aware bounded queue sink for loguru.

    This sink wraps any callable sink and provides memory safety by using
    a bounded queue with configurable overflow policies. A background worker
    thread consumes messages from the queue and forwards to the actual sink.

    Args:
        sink: The actual sink callable (e.g., file.write, print)
        maxsize: Maximum queue size before overflow handling kicks in
        drop_policy: How to handle queue overflow (default: OLDEST)
        name: Optional name for the worker thread (debugging)

    Example:
        >>> from loguru import logger
        >>> import sys
        >>>
        >>> def file_sink(msg: str) -> None:
        ...     with open("app.log", "a") as f:
        ...         f.write(msg)
        ...
        >>> bounded = BoundedQueueSink(file_sink, maxsize=5000)
        >>> logger.add(bounded.write, format="{message}")
    """

    def __init__(
        self,
        sink: Callable[[str], None],
        maxsize: int = 10000,
        drop_policy: DropPolicy = DropPolicy.OLDEST,
        name: str = "BoundedQueueWorker",
    ) -> None:
        """Initialize the bounded queue sink.

        Args:
            sink: Target sink callable
            maxsize: Queue capacity (default 10000 messages)
            drop_policy: Overflow handling strategy
            name: Worker thread name
        """
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=maxsize)
        self._sink = sink
        self._drop_policy = drop_policy
        self._dropped_count = 0
        self._lock = threading.Lock()
        self._running = True

        # Start background consumer thread
        self._worker = threading.Thread(
            target=self._consume,
            name=name,
            daemon=True,
        )
        self._worker.start()

        # Register cleanup on interpreter shutdown
        atexit.register(self.stop)

    def write(self, message: str) -> None:
        """Write a message to the queue (non-blocking).

        This method is called by loguru for each log record.
        It never blocks the caller in OLDEST/NEWEST modes.

        Args:
            message: Formatted log message string
        """
        if not self._running:
            return

        try:
            if self._drop_policy == DropPolicy.BLOCK:
                self._queue.put(message)  # Blocking put
            else:
                self._queue.put_nowait(message)
        except queue.Full:
            self._handle_overflow(message)

    def _handle_overflow(self, message: str) -> None:
        """Handle queue overflow according to drop policy.

        Args:
            message: The message that couldn't be enqueued
        """
        with self._lock:
            self._dropped_count += 1

        if self._drop_policy == DropPolicy.OLDEST:
            # Drop oldest to make room for newest
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(message)
            except (queue.Empty, queue.Full):
                pass  # Race condition, message dropped
        # NEWEST policy: silently drop the incoming message (do nothing)

    def _consume(self) -> None:
        """Background worker that consumes queue and forwards to sink."""
        while self._running:
            try:
                message = self._queue.get(timeout=1.0)
                if message is None:
                    break
                self._sink(message)
            except queue.Empty:
                continue
            except Exception:
                # Sink error - log to stderr but don't crash worker
                print("[BoundedQueueSink] Sink error", file=sys.stderr)

    def stop(self) -> None:
        """Gracefully stop the worker thread.

        Drains remaining messages before stopping.
        Called automatically at interpreter shutdown via atexit.
        """
        if not self._running:
            return

        self._running = False
        self._queue.put(None)  # Sentinel to wake up worker

        # Wait for worker to finish (with timeout)
        self._worker.join(timeout=5.0)

    @property
    def dropped_count(self) -> int:
        """Number of messages dropped due to queue overflow."""
        with self._lock:
            return self._dropped_count

    @property
    def queue_size(self) -> int:
        """Current number of messages in queue."""
        return self._queue.qsize()


def create_bounded_file_sink(
    path: str,
    maxsize: int = 10000,
    drop_policy: DropPolicy = DropPolicy.OLDEST,
) -> BoundedQueueSink:
    """Factory function to create a bounded file sink.

    Convenience wrapper that creates a BoundedQueueSink with a file writer.

    Args:
        path: Path to log file
        maxsize: Queue capacity
        drop_policy: Overflow handling strategy

    Returns:
        Configured BoundedQueueSink instance
    """

    file_path = Path(path)

    def file_writer(message: str) -> None:
        with file_path.open("a", encoding="utf-8") as f:
            f.write(message)

    return BoundedQueueSink(
        sink=file_writer,
        maxsize=maxsize,
        drop_policy=drop_policy,
        name=f"FileWriter-{path}",
    )
