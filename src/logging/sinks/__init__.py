"""Custom sink implementations for loguru.

This package provides production-ready sinks that solve common issues:
- BoundedQueueSink: Prevents OOM from slow sinks (loguru Issue #1419)
- DiscordWebhookSink: Async alerting for ERROR+ logs
- OpenTelemetry integration: trace_id/span_id injection
"""

from src.logging.sinks.bounded_queue import BoundedQueueSink
from src.logging.sinks.discord import DiscordWebhookSink

__all__ = ["BoundedQueueSink", "DiscordWebhookSink"]
