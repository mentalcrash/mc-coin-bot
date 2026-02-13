"""거래소 API 메트릭 콜백 테스트 — ApiMetricsCallback Protocol + 계측 검증."""

from __future__ import annotations

from prometheus_client import REGISTRY

from src.monitoring.metrics import (
    ApiMetricsCallback,
    PrometheusApiCallback,
    PrometheusWsCallback,
    WsStatusCallback,
)


def _sample(name: str, labels: dict[str, str] | None = None) -> float | None:
    """REGISTRY에서 특정 metric 값을 가져옴."""
    return REGISTRY.get_sample_value(name, labels or {})


class TestApiMetricsCallbackProtocol:
    def test_prometheus_callback_satisfies_protocol(self) -> None:
        """PrometheusApiCallback은 ApiMetricsCallback Protocol을 만족해야 함."""
        cb = PrometheusApiCallback()
        assert isinstance(cb, ApiMetricsCallback)

    def test_prometheus_ws_callback_satisfies_protocol(self) -> None:
        """PrometheusWsCallback은 WsStatusCallback Protocol을 만족해야 함."""
        cb = PrometheusWsCallback()
        assert isinstance(cb, WsStatusCallback)


class TestPrometheusApiCallbackMetrics:
    def test_success_increments_counter(self) -> None:
        cb = PrometheusApiCallback()
        before = (
            _sample(
                "mcbot_exchange_api_calls_total",
                {"endpoint": "fetch_balance", "status": "success"},
            )
            or 0.0
        )

        cb.on_api_call("fetch_balance", 0.25, "success")

        after = _sample(
            "mcbot_exchange_api_calls_total",
            {"endpoint": "fetch_balance", "status": "success"},
        )
        assert after is not None
        assert after > before

    def test_failure_increments_counter(self) -> None:
        cb = PrometheusApiCallback()
        before = (
            _sample(
                "mcbot_exchange_api_calls_total",
                {"endpoint": "fetch_positions", "status": "failure"},
            )
            or 0.0
        )

        cb.on_api_call("fetch_positions", 1.5, "failure")

        after = _sample(
            "mcbot_exchange_api_calls_total",
            {"endpoint": "fetch_positions", "status": "failure"},
        )
        assert after is not None
        assert after > before

    def test_retry_increments_counter(self) -> None:
        cb = PrometheusApiCallback()
        before = (
            _sample(
                "mcbot_exchange_api_calls_total",
                {"endpoint": "create_order", "status": "retry"},
            )
            or 0.0
        )

        cb.on_api_call("create_order", 2.0, "retry")

        after = _sample(
            "mcbot_exchange_api_calls_total",
            {"endpoint": "create_order", "status": "retry"},
        )
        assert after is not None
        assert after > before

    def test_latency_histogram_recorded(self) -> None:
        cb = PrometheusApiCallback()
        # Histogram sum 확인
        before = _sample("mcbot_exchange_api_latency_seconds_sum", {"endpoint": "fetch_ticker"})
        before = before or 0.0

        cb.on_api_call("fetch_ticker", 0.35, "success")

        after = _sample("mcbot_exchange_api_latency_seconds_sum", {"endpoint": "fetch_ticker"})
        assert after is not None
        assert after >= before + 0.35


class TestPrometheusWsCallbackMetrics:
    def test_connected_sets_gauge_to_1(self) -> None:
        cb = PrometheusWsCallback()
        cb.on_ws_status("SOL/USDT", connected=True)
        val = _sample("mcbot_exchange_ws_connected", {"symbol": "SOL/USDT"})
        assert val == 1.0

    def test_disconnected_sets_gauge_to_0(self) -> None:
        cb = PrometheusWsCallback()
        cb.on_ws_status("DOGE/USDT", connected=False)
        val = _sample("mcbot_exchange_ws_connected", {"symbol": "DOGE/USDT"})
        assert val == 0.0

    def test_reconnection_updates_gauge(self) -> None:
        cb = PrometheusWsCallback()
        cb.on_ws_status("XRP/USDT", connected=True)
        assert _sample("mcbot_exchange_ws_connected", {"symbol": "XRP/USDT"}) == 1.0

        cb.on_ws_status("XRP/USDT", connected=False)
        assert _sample("mcbot_exchange_ws_connected", {"symbol": "XRP/USDT"}) == 0.0

        cb.on_ws_status("XRP/USDT", connected=True)
        assert _sample("mcbot_exchange_ws_connected", {"symbol": "XRP/USDT"}) == 1.0
