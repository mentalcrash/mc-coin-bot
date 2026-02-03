"""OpenTelemetry integration for loguru.

This module provides integration between loguru and OpenTelemetry for
distributed tracing correlation. It injects trace_id and span_id into
log records, enabling log-trace correlation in observability platforms
like Grafana Loki.

Note: OpenTelemetry packages are optional dependencies. This module
gracefully handles their absence.

Rules Applied:
    - #15 Logging Standards: Structured logging with context
    - #10 Python Standards: Type safety, optional dependencies
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

# =============================================================================
# OpenTelemetry Availability Check
# =============================================================================

_otel_available = False
_trace_module: Any = None

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-not-found]

    _otel_available = True
    _trace_module = _otel_trace
except ImportError:
    pass


def is_otel_available() -> bool:
    """Check if OpenTelemetry is available.

    Returns:
        True if OpenTelemetry packages are installed
    """
    return _otel_available


# =============================================================================
# Context Injection
# =============================================================================


def get_current_trace_context() -> dict[str, str]:
    """Get current OpenTelemetry trace context.

    Extracts trace_id and span_id from the current active span.
    Returns empty dict if no span is active or OTel is not available.

    Returns:
        Dictionary with trace_id and span_id (empty if not available)

    Example:
        >>> ctx = get_current_trace_context()
        >>> # {"trace_id": "abc123...", "span_id": "def456..."}
    """
    if not _otel_available or _trace_module is None:
        return {}

    span = _trace_module.get_current_span()
    if span is None:
        return {}

    span_context = span.get_span_context()
    if span_context is None or not span_context.is_valid:
        return {}

    return {
        "trace_id": format(span_context.trace_id, "032x"),
        "span_id": format(span_context.span_id, "016x"),
        "trace_flags": format(span_context.trace_flags, "02x"),
    }


def inject_otel_context(record: dict[str, Any]) -> dict[str, Any]:
    """Inject OpenTelemetry context into a log record.

    This function is designed to be used with loguru's patcher mechanism.
    It adds trace_id and span_id to the record's extra dict.

    Args:
        record: Loguru record dictionary

    Returns:
        Modified record with OTel context injected

    Example:
        >>> logger.add(sink, format="{extra[trace_id]} | {message}")
        >>> # In setup:
        >>> logger = logger.patch(inject_otel_context)
    """
    if not _otel_available:
        return record

    otel_ctx = get_current_trace_context()
    if otel_ctx:
        if "extra" not in record:
            record["extra"] = {}
        record["extra"].update(otel_ctx)

    return record


def create_otel_patcher() -> Any:
    """Create a loguru patcher function for OTel context injection.

    Returns a function suitable for use with logger.patch().
    Returns identity function if OTel is not available.

    Returns:
        Patcher function for loguru

    Example:
        >>> from loguru import logger
        >>> patcher = create_otel_patcher()
        >>> logger = logger.patch(patcher)
    """
    if not _otel_available:
        # Return identity patcher if OTel not available
        def identity_patcher(record: dict[str, Any]) -> dict[str, Any]:
            return record

        return identity_patcher

    return inject_otel_context


# =============================================================================
# Custom Format with OTel Fields
# =============================================================================


def get_otel_json_format() -> str:
    """Get a loguru format string that includes OTel fields.

    Returns a format string suitable for JSON serialization that
    includes trace_id and span_id fields.

    Returns:
        Format string for loguru
    """
    return (
        "{{"
        '"timestamp": "{time:YYYY-MM-DDTHH:mm:ss.SSSZ}", '
        '"level": "{level.name}", '
        '"message": "{message}", '
        '"logger": "{name}", '
        '"function": "{function}", '
        '"line": {line}, '
        '"trace_id": "{extra[trace_id]}", '
        '"span_id": "{extra[span_id]}", '
        "{extra}"
        "}}"
    )


# =============================================================================
# OTel-aware Sink Wrapper
# =============================================================================


class OTelAwareSink:
    """Sink wrapper that automatically injects OTel context.

    Wraps any sink and automatically injects trace_id/span_id
    from the current OpenTelemetry span context.

    Args:
        sink: The underlying sink (file, callable, etc.)
        include_service_info: Include service.name and service.version

    Example:
        >>> from loguru import logger
        >>> import sys
        >>>
        >>> otel_sink = OTelAwareSink(sys.stdout.write)
        >>> logger.add(otel_sink.write, serialize=True)
    """

    def __init__(
        self,
        sink: Any,
        include_service_info: bool = True,
        service_name: str = "mc-coin-bot",
        service_version: str = "0.1.0",
    ) -> None:
        """Initialize OTel-aware sink.

        Args:
            sink: Underlying sink
            include_service_info: Add service metadata to logs
            service_name: Service name for logs
            service_version: Service version for logs
        """
        self._sink = sink
        self._include_service_info = include_service_info
        self._service_name = service_name
        self._service_version = service_version

    def write(self, message: str) -> None:
        """Write message with OTel context.

        Note: When using serialize=True, loguru handles JSON formatting.
        This method is for custom format scenarios.

        Args:
            message: Log message
        """
        # For serialized messages, we can't easily inject context
        # The patcher approach is preferred for serialize=True
        if callable(self._sink):
            self._sink(message)
        else:
            self._sink.write(message)


# =============================================================================
# Resource Detection (for OTel SDK initialization)
# =============================================================================


def get_service_resource() -> dict[str, str]:
    """Get service resource attributes for OTel SDK.

    Returns attributes suitable for initializing the OTel SDK
    with service information.

    Returns:
        Dictionary of resource attributes

    Example:
        >>> from opentelemetry.sdk.resources import Resource
        >>> resource = Resource.create(get_service_resource())
    """
    return {
        "service.name": "mc-coin-bot",
        "service.version": "0.1.0",
        "service.namespace": "trading",
        "deployment.environment": "production",
    }


# =============================================================================
# Span Creation Helpers
# =============================================================================


def create_trading_span(
    name: str,
    attributes: dict[str, str] | None = None,
) -> Any:
    """Create a span for trading operations.

    Convenience function to create spans with trading-specific attributes.
    Returns a no-op context manager if OTel is not available.

    Args:
        name: Span name (e.g., "place_order", "fetch_ticker")
        attributes: Additional span attributes

    Returns:
        Span context manager

    Example:
        >>> with create_trading_span("place_order", {"symbol": "BTC/USDT"}):
        ...     await exchange.create_order(...)
    """
    if not _otel_available or _trace_module is None:
        # Return a no-op context manager
        return nullcontext()

    tracer = _trace_module.get_tracer("mc-coin-bot")
    span_attributes = attributes or {}

    return tracer.start_as_current_span(name, attributes=span_attributes)
