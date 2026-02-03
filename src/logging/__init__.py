"""Logging service module for high-performance trading bot.

This module provides a production-ready logging infrastructure with:
- Bounded queue sinks for backpressure handling (memory safety)
- Discord webhook integration for alerting
- OpenTelemetry context injection for observability
- Context binding utilities for async environments

Rules Applied:
    - #15 Logging Standards: Loguru, dual sinks, enqueue alternatives
    - #22 Notification Standards: Discord webhooks, Rich embeds
    - #23 Exception Handling: Alert hooks for critical errors
"""

from src.logging.config import LoggingConfig, get_logging_config
from src.logging.context import (
    LoggingContext,
    clear_context,
    generate_order_id,
    generate_trace_id,
    get_current_context,
    get_execution_logger,
    get_order_logger,
    get_strategy_logger,
    get_trading_logger,
)

__all__ = [
    "LoggingConfig",
    "LoggingContext",
    "clear_context",
    "generate_order_id",
    "generate_trace_id",
    "get_current_context",
    "get_execution_logger",
    "get_logging_config",
    "get_order_logger",
    "get_strategy_logger",
    "get_trading_logger",
]
