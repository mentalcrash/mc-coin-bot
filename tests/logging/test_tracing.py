"""Tests for OTel Full Tracing module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.logging.tracing import (
    TraceContextStore,
    _NoOpSpan,
    _NoOpTracer,
    component_span,
    component_span_with_context,
    get_tracer,
    trade_cycle_span,
)


class TestNoOpWhenUnavailable:
    """OTel 미설치 시 no-op 동작 검증."""

    def test_get_tracer_returns_noop(self) -> None:
        """OTel unavailable → _NoOpTracer."""
        with patch("src.logging.tracing._otel_available", False):
            tracer = get_tracer()
            assert isinstance(tracer, _NoOpTracer)

    def test_noop_tracer_span(self) -> None:
        """NoOpTracer.start_as_current_span → NoOpSpan."""
        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test")
        assert isinstance(span, _NoOpSpan)

    def test_noop_span_methods(self) -> None:
        """NoOpSpan의 메서드가 에러 없이 실행."""
        span = _NoOpSpan()
        span.set_attribute("key", "value")
        span.set_status()
        span.record_exception(ValueError("test"))
        # context manager
        with span:
            pass

    def test_trade_cycle_span_noop(self) -> None:
        """OTel unavailable → trade_cycle_span이 None yield."""
        with patch("src.logging.tracing._otel_available", False):
            with trade_cycle_span("BTC/USDT", "tsmom", "abc-123") as span:
                assert span is None

    def test_component_span_noop(self) -> None:
        """OTel unavailable → component_span이 None yield."""
        with patch("src.logging.tracing._otel_available", False):
            with component_span("pm.process_signal") as span:
                assert span is None

    def test_component_span_with_context_noop(self) -> None:
        """OTel unavailable → component_span_with_context가 None yield."""
        with patch("src.logging.tracing._otel_available", False):
            with component_span_with_context("rm.check", "corr-id") as span:
                assert span is None


class TestTraceContextStore:
    """TraceContextStore — store/retrieve, LRU eviction."""

    def test_store_and_retrieve(self) -> None:
        store = TraceContextStore(max_size=100)
        ctx = MagicMock()
        store.store("id-1", ctx)
        assert store.retrieve("id-1") is ctx

    def test_retrieve_missing_returns_none(self) -> None:
        store = TraceContextStore(max_size=100)
        assert store.retrieve("nonexistent") is None

    def test_lru_eviction(self) -> None:
        """max_size 초과 시 가장 오래된 항목 제거."""
        store = TraceContextStore(max_size=3)
        store.store("a", "ctx-a")
        store.store("b", "ctx-b")
        store.store("c", "ctx-c")
        store.store("d", "ctx-d")  # evicts "a"

        assert store.retrieve("a") is None
        assert store.retrieve("b") == "ctx-b"
        assert store.retrieve("d") == "ctx-d"
        assert len(store) == 3

    def test_retrieve_refreshes_order(self) -> None:
        """retrieve 시 LRU 순서 갱신."""
        store = TraceContextStore(max_size=3)
        store.store("a", "ctx-a")
        store.store("b", "ctx-b")
        store.store("c", "ctx-c")

        # "a" 조회 → LRU 순서에서 최근으로
        store.retrieve("a")

        # "d" 추가 → "b"가 evict (가장 오래된)
        store.store("d", "ctx-d")
        assert store.retrieve("b") is None
        assert store.retrieve("a") == "ctx-a"

    def test_update_existing_key(self) -> None:
        """같은 key로 store → 값 업데이트."""
        store = TraceContextStore(max_size=10)
        store.store("a", "old")
        store.store("a", "new")
        assert store.retrieve("a") == "new"
        assert len(store) == 1

    def test_len(self) -> None:
        store = TraceContextStore(max_size=100)
        assert len(store) == 0
        store.store("a", "ctx")
        assert len(store) == 1


class TestSpanCreation:
    """Span 생성 테스트 (mock OTel)."""

    def test_trade_cycle_span_creates_span(self) -> None:
        """OTel available 시 span이 생성."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.logging.tracing._otel_available", True),
            patch("src.logging.tracing.get_tracer", return_value=mock_tracer),
            patch("src.logging.tracing._trace_api", None),
        ):
            with trade_cycle_span("BTC/USDT", "tsmom", "corr-123") as span:
                assert span is mock_span

    def test_component_span_creates_span(self) -> None:
        """OTel available 시 component span 생성."""
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("src.logging.tracing._otel_available", True),
            patch("src.logging.tracing.get_tracer", return_value=mock_tracer),
        ):
            with component_span("pm.process_signal", {"symbol": "BTC/USDT"}) as span:
                assert span is mock_span


class TestSetupTracing:
    """setup_tracing 테스트."""

    def test_setup_skipped_when_unavailable(self) -> None:
        """OTel 미설치 시 setup이 무시됨."""
        with patch("src.logging.tracing._otel_available", False):
            # 에러 없이 실행
            from src.logging.tracing import setup_tracing

            setup_tracing()


class TestContextPropagation:
    """correlation_id 기반 context 전파."""

    def test_store_then_retrieve_context(self) -> None:
        """trade_cycle_span에서 저장 → component_span_with_context에서 조회."""
        store = TraceContextStore(max_size=100)
        mock_ctx = MagicMock()
        store.store("corr-abc", mock_ctx)
        retrieved = store.retrieve("corr-abc")
        assert retrieved is mock_ctx
