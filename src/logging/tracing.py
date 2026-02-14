"""OTel Full Tracing — 주문 lifecycle span 관리.

EDA 컴포넌트 간 trace context를 전파하여
StrategyEngine → PM → RM → OMS → Executor 전체 lifecycle을 추적합니다.

Span propagation: correlation_id → TraceContextStore 기반.
기존 이벤트 모델에 correlation_id가 이미 존재하므로 이벤트 변경 불필요.

OpenTelemetry SDK는 optional dependency — 미설치 시 모든 함수가 no-op.

Rules Applied:
    - Optional import pattern (otel.py 동일)
    - No-op when unavailable
    - Bounded LRU for context store
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from typing import Any, Generator

from loguru import logger

# =============================================================================
# OpenTelemetry Availability Check
# =============================================================================

_otel_available = False
_trace_module: Any = None
_trace_api: Any = None

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-not-found]
    from opentelemetry import context as _otel_context  # type: ignore[import-not-found]

    _otel_available = True
    _trace_module = _otel_trace
    _trace_api = _otel_context
except ImportError:
    pass

_provider_initialized = False


# =============================================================================
# Setup / Shutdown
# =============================================================================


def setup_tracing(
    service_name: str = "mc-coin-bot",
    endpoint: str | None = None,
    *,
    console_export: bool = False,
) -> None:
    """TracerProvider 초기화. LiveRunner.run() 시작 시 호출.

    Args:
        service_name: OTel service name
        endpoint: OTLP exporter endpoint (None이면 exporter 없음)
        console_export: True면 ConsoleSpanExporter 추가 (디버깅용)
    """
    global _provider_initialized  # noqa: PLW0603

    if not _otel_available:
        logger.debug("OTel not available, tracing setup skipped")
        return

    if _provider_initialized:
        return

    try:
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-not-found]

            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        if console_export:
            from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
                ConsoleSpanExporter,
                SimpleSpanProcessor,
            )

            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        _trace_module.set_tracer_provider(provider)
        _provider_initialized = True
        logger.info("OTel tracing initialized: service={}, endpoint={}", service_name, endpoint)
    except Exception:
        logger.opt(exception=True).warning("OTel tracing setup failed, continuing without tracing")


def shutdown_tracing() -> None:
    """TracerProvider flush + shutdown. Graceful shutdown 시 호출."""
    global _provider_initialized  # noqa: PLW0603

    if not _otel_available or not _provider_initialized:
        return

    try:
        provider = _trace_module.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        _provider_initialized = False
        logger.info("OTel tracing shutdown complete")
    except Exception:
        logger.opt(exception=True).warning("OTel tracing shutdown error")


# =============================================================================
# Tracer Access
# =============================================================================


def get_tracer(name: str = "mc-coin-bot") -> Any:
    """Tracer 인스턴스. OTel 미설치 시 no-op tracer.

    Args:
        name: tracer name

    Returns:
        OTel Tracer or _NoOpTracer
    """
    if not _otel_available or _trace_module is None:
        return _NoOpTracer()
    return _trace_module.get_tracer(name)


# =============================================================================
# Span Context Managers
# =============================================================================


@contextmanager
def trade_cycle_span(
    symbol: str,
    strategy: str,
    correlation_id: str | None = None,
) -> Generator[Any, None, None]:
    """주문 lifecycle root span.

    StrategyEngine._on_bar()에서 시그널 생성 시 root span을 시작하고,
    correlation_id로 TraceContextStore에 context를 저장합니다.

    Args:
        symbol: 거래 심볼
        strategy: 전략 이름
        correlation_id: 이벤트 correlation_id (UUID 문자열)

    Yields:
        span 인스턴스 (OTel 미설치 시 None)
    """
    if not _otel_available:
        yield None
        return

    tracer = get_tracer()
    attrs = {"symbol": symbol, "strategy": strategy}
    if correlation_id:
        attrs["correlation_id"] = correlation_id

    with tracer.start_as_current_span("trade_cycle", attributes=attrs) as span:
        # correlation_id가 있으면 context 저장
        if correlation_id and _trace_api is not None:
            ctx = _trace_api.get_current()
            _trace_context_store.store(correlation_id, ctx)
        yield span


@contextmanager
def component_span(
    name: str,
    attributes: dict[str, str | float | int] | None = None,
) -> Generator[Any, None, None]:
    """컴포넌트별 child span.

    PM, RM, OMS, Executor에서 correlation_id를 통해
    parent context를 복원한 뒤 child span을 생성합니다.

    Args:
        name: span 이름 (e.g. "pm.process_signal")
        attributes: span attributes

    Yields:
        span 인스턴스 (OTel 미설치 시 None)
    """
    if not _otel_available:
        yield None
        return

    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        yield span


@contextmanager
def component_span_with_context(
    name: str,
    correlation_id: str | None = None,
    attributes: dict[str, str | float | int] | None = None,
) -> Generator[Any, None, None]:
    """Parent context를 복원한 뒤 child span 생성.

    correlation_id로 TraceContextStore에서 parent context를 조회하여
    해당 context 아래에 child span을 생성합니다.

    Args:
        name: span 이름
        correlation_id: 이벤트 correlation_id (UUID 문자열)
        attributes: span attributes

    Yields:
        span 인스턴스 (OTel 미설치 시 None)
    """
    if not _otel_available:
        yield None
        return

    tracer = get_tracer()
    ctx = None
    if correlation_id:
        ctx = _trace_context_store.retrieve(correlation_id)

    if ctx is not None and _trace_api is not None:
        token = _trace_api.attach(ctx)
        try:
            with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
                yield span
        finally:
            _trace_api.detach(token)
    else:
        with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
            yield span


# =============================================================================
# TraceContextStore
# =============================================================================

_MAX_CONTEXT_STORE_SIZE = 10000


class TraceContextStore:
    """correlation_id → OTel context 매핑 (bounded LRU).

    최대 크기 초과 시 가장 오래된 항목부터 제거합니다.

    Args:
        max_size: 최대 저장 크기 (default 10000)
    """

    __slots__ = ("_max_size", "_store")

    def __init__(self, max_size: int = _MAX_CONTEXT_STORE_SIZE) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size

    def store(self, correlation_id: str, context: Any) -> None:
        """Context 저장.

        Args:
            correlation_id: 이벤트 correlation_id
            context: OTel context 객체
        """
        if correlation_id in self._store:
            self._store.move_to_end(correlation_id)
            self._store[correlation_id] = context
        else:
            self._store[correlation_id] = context
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def retrieve(self, correlation_id: str) -> Any:
        """Context 조회.

        Args:
            correlation_id: 이벤트 correlation_id

        Returns:
            OTel context 또는 None
        """
        ctx = self._store.get(correlation_id)
        if ctx is not None:
            self._store.move_to_end(correlation_id)
        return ctx

    def __len__(self) -> int:
        return len(self._store)


# Module-level singleton
_trace_context_store = TraceContextStore()


# =============================================================================
# No-Op Fallback
# =============================================================================


class _NoOpSpan:
    """OTel 미설치 시 span 대체."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """OTel 미설치 시 tracer 대체."""

    def start_as_current_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _NoOpSpan:
        return _NoOpSpan()
