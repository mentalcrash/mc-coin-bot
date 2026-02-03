"""Context binding utilities for structured logging.

This module provides async-safe context propagation using contextvars.
It ensures that order_id, strategy, and trace_id are properly attached
to all log records even in concurrent async environments.

Rules Applied:
    - #15 Logging Standards: Context binding with logger.bind()
    - #10 Python Standards: contextvars for async safety
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# =============================================================================
# Context Variables (Async-Safe)
# =============================================================================
# These are automatically propagated across await boundaries in asyncio

current_order_id: ContextVar[str | None] = ContextVar("order_id", default=None)
current_strategy: ContextVar[str | None] = ContextVar("strategy", default=None)
current_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
current_symbol: ContextVar[str | None] = ContextVar("symbol", default=None)
current_exchange: ContextVar[str | None] = ContextVar("exchange", default=None)


# =============================================================================
# Logger Factory Functions
# =============================================================================


def get_trading_logger(
    *,
    order_id: str | None = None,
    strategy: str | None = None,
    trace_id: str | None = None,
    symbol: str | None = None,
    exchange: str | None = None,
    **extra: str,
) -> Logger:
    """Get a logger with trading context bound.

    Creates a new logger instance with the provided context values bound.
    These values will be included in all log records from this logger,
    enabling structured logging and log correlation.

    Args:
        order_id: Unique order identifier for tracking order lifecycle
        strategy: Strategy name (e.g., "TSMOM", "MeanReversion")
        trace_id: OpenTelemetry trace ID for distributed tracing
        symbol: Trading symbol (e.g., "BTC/USDT")
        exchange: Exchange name (e.g., "binance")
        **extra: Additional context key-value pairs

    Returns:
        Logger instance with context bound

    Example:
        >>> log = get_trading_logger(
        ...     order_id="ord_123",
        ...     strategy="TSMOM",
        ...     symbol="BTC/USDT"
        ... )
        >>> log.info("Order placed")
        # Output includes: {"order_id": "ord_123", "strategy": "TSMOM", ...}
    """
    ctx: dict[str, str] = {}

    # Set context variables and build bind dict
    if order_id:
        ctx["order_id"] = order_id
        current_order_id.set(order_id)
    if strategy:
        ctx["strategy"] = strategy
        current_strategy.set(strategy)
    if trace_id:
        ctx["trace_id"] = trace_id
        current_trace_id.set(trace_id)
    if symbol:
        ctx["symbol"] = symbol
        current_symbol.set(symbol)
    if exchange:
        ctx["exchange"] = exchange
        current_exchange.set(exchange)

    ctx.update(extra)

    return logger.bind(**ctx)


def get_order_logger(
    order_id: str,
    symbol: str,
    strategy: str | None = None,
) -> Logger:
    """Get a logger specifically for order lifecycle tracking.

    Convenience function for order-related logging with required fields.

    Args:
        order_id: Unique order identifier
        symbol: Trading symbol
        strategy: Strategy name (optional)

    Returns:
        Logger bound with order context

    Example:
        >>> log = get_order_logger("ord_abc123", "ETH/USDT", "TSMOM")
        >>> log.info("Order submitted to exchange")
    """
    return get_trading_logger(
        order_id=order_id,
        symbol=symbol,
        strategy=strategy,
    )


def get_strategy_logger(
    strategy: str,
    symbol: str | None = None,
) -> Logger:
    """Get a logger for strategy-level logging.

    Args:
        strategy: Strategy name
        symbol: Trading symbol (optional, for symbol-specific strategy logs)

    Returns:
        Logger bound with strategy context

    Example:
        >>> log = get_strategy_logger("TSMOM")
        >>> log.info("Strategy initialized")
    """
    return get_trading_logger(strategy=strategy, symbol=symbol)


def get_execution_logger(
    order_id: str | None = None,
    trace_id: str | None = None,
) -> Logger:
    """Get a logger for execution layer operations.

    Automatically generates a trace_id if not provided for
    request correlation.

    Args:
        order_id: Order being executed (if applicable)
        trace_id: Trace ID for distributed tracing (auto-generated if None)

    Returns:
        Logger bound with execution context
    """
    if trace_id is None:
        trace_id = generate_trace_id()

    return get_trading_logger(order_id=order_id, trace_id=trace_id)


# =============================================================================
# Utility Functions
# =============================================================================


def generate_trace_id() -> str:
    """Generate a unique trace ID for request correlation.

    Returns:
        32-character hex string (compatible with OpenTelemetry format)
    """
    return uuid.uuid4().hex


def generate_order_id(prefix: str = "ord") -> str:
    """Generate a unique order ID.

    Args:
        prefix: Prefix for the order ID (default: "ord")

    Returns:
        Unique order identifier (e.g., "ord_a1b2c3d4")
    """
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}_{short_uuid}"


def get_current_context() -> dict[str, str | None]:
    """Get all current context values.

    Useful for debugging or passing context to other systems.

    Returns:
        Dictionary of current context values
    """
    return {
        "order_id": current_order_id.get(),
        "strategy": current_strategy.get(),
        "trace_id": current_trace_id.get(),
        "symbol": current_symbol.get(),
        "exchange": current_exchange.get(),
    }


def clear_context() -> None:
    """Clear all context variables.

    Should be called at the start of new request/task processing
    to prevent context leakage.
    """
    current_order_id.set(None)
    current_strategy.set(None)
    current_trace_id.set(None)
    current_symbol.set(None)
    current_exchange.set(None)


# =============================================================================
# Context Manager for Scoped Logging
# =============================================================================


class LoggingContext:
    """Context manager for scoped logging context.

    Automatically sets and clears context variables within a scope.
    Useful for ensuring context is properly cleaned up.

    Example:
        >>> async with LoggingContext(order_id="123", strategy="TSMOM"):
        ...     logger.info("Processing order")  # Includes context
        >>> logger.info("After scope")  # Context cleared
    """

    def __init__(
        self,
        order_id: str | None = None,
        strategy: str | None = None,
        trace_id: str | None = None,
        symbol: str | None = None,
        exchange: str | None = None,
    ) -> None:
        """Initialize logging context.

        Args:
            order_id: Order ID to set
            strategy: Strategy name to set
            trace_id: Trace ID to set
            symbol: Symbol to set
            exchange: Exchange to set
        """
        self._order_id = order_id
        self._strategy = strategy
        self._trace_id = trace_id
        self._symbol = symbol
        self._exchange = exchange
        self._tokens: dict[str, object] = {}

    def __enter__(self) -> LoggingContext:
        """Enter context and set variables."""
        if self._order_id:
            self._tokens["order_id"] = current_order_id.set(self._order_id)
        if self._strategy:
            self._tokens["strategy"] = current_strategy.set(self._strategy)
        if self._trace_id:
            self._tokens["trace_id"] = current_trace_id.set(self._trace_id)
        if self._symbol:
            self._tokens["symbol"] = current_symbol.set(self._symbol)
        if self._exchange:
            self._tokens["exchange"] = current_exchange.set(self._exchange)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context and reset variables."""
        # Map key names to actual ContextVar objects
        context_vars = {
            "order_id": current_order_id,
            "strategy": current_strategy,
            "trace_id": current_trace_id,
            "symbol": current_symbol,
            "exchange": current_exchange,
        }
        for key, token in self._tokens.items():
            context_vars[key].reset(token)  # type: ignore[arg-type]

    async def __aenter__(self) -> LoggingContext:
        """Async enter - delegates to sync enter."""
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async exit - delegates to sync exit."""
        self.__exit__(exc_type, exc_val, exc_tb)
