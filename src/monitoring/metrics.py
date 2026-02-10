"""MetricsExporter — Prometheus 메트릭 수출기.

EventBus subscriber로 실시간 메트릭을 수집하고,
prometheus_client HTTP 서버를 통해 /metrics endpoint로 노출합니다.

Rules Applied:
    - EDA 패턴: EventBus subscribe
    - Prometheus naming: mcbot_ prefix
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger
from prometheus_client import Counter, Gauge

from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    BarEvent,
    CircuitBreakerEvent,
    EventType,
    FillEvent,
    PositionUpdateEvent,
    RiskAlertEvent,
    SignalEvent,
)

if TYPE_CHECKING:
    from src.core.event_bus import EventBus

# ==========================================================================
# Gauges
# ==========================================================================
equity_gauge = Gauge("mcbot_equity_usdt", "Current account equity")
drawdown_gauge = Gauge("mcbot_drawdown_pct", "Current drawdown percentage")
cash_gauge = Gauge("mcbot_cash_usdt", "Available cash")
position_count_gauge = Gauge("mcbot_open_positions", "Number of open positions")
position_size_gauge = Gauge("mcbot_position_size", "Position size", ["symbol"])
uptime_gauge = Gauge("mcbot_uptime_seconds", "Bot uptime in seconds")

# ==========================================================================
# Counters
# ==========================================================================
fills_counter = Counter("mcbot_fills", "Total fills executed", ["symbol", "side"])
signals_counter = Counter("mcbot_signals", "Total signals generated", ["symbol"])
bars_counter = Counter("mcbot_bars", "Total bars processed", ["timeframe"])
cb_triggered_counter = Counter("mcbot_circuit_breaker", "Circuit breaker activations")
risk_alerts_counter = Counter("mcbot_risk_alerts", "Risk alerts", ["level"])


class MetricsExporter:
    """Prometheus 메트릭 수출기 — EventBus subscriber.

    Args:
        port: Prometheus HTTP 서버 포트 (0=비활성)
    """

    def __init__(self, port: int = 8000) -> None:
        self._port = port
        self._start_time = time.monotonic()

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록.

        Args:
            bus: EventBus 인스턴스
        """
        bus.subscribe(EventType.BALANCE_UPDATE, self._on_balance)
        bus.subscribe(EventType.FILL, self._on_fill)
        bus.subscribe(EventType.SIGNAL, self._on_signal)
        bus.subscribe(EventType.BAR, self._on_bar)
        bus.subscribe(EventType.CIRCUIT_BREAKER, self._on_cb)
        bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert)
        bus.subscribe(EventType.POSITION_UPDATE, self._on_position)
        logger.info("MetricsExporter registered to EventBus")

    def start_server(self) -> None:
        """Prometheus HTTP 서버 시작 (:port/metrics)."""
        if self._port <= 0:
            logger.info("Prometheus metrics server disabled (port=0)")
            return
        from prometheus_client import start_http_server

        start_http_server(self._port)
        logger.info("Prometheus metrics server started on port {}", self._port)

    def update_uptime(self) -> None:
        """Uptime gauge 갱신."""
        uptime_gauge.set(time.monotonic() - self._start_time)

    async def _on_balance(self, event: AnyEvent) -> None:
        """BalanceUpdateEvent → equity/cash/drawdown gauges."""
        assert isinstance(event, BalanceUpdateEvent)
        equity_gauge.set(event.total_equity)
        cash_gauge.set(event.available_cash)

    async def _on_fill(self, event: AnyEvent) -> None:
        """FillEvent → fills counter."""
        assert isinstance(event, FillEvent)
        fills_counter.labels(symbol=event.symbol, side=event.side).inc()

    async def _on_signal(self, event: AnyEvent) -> None:
        """SignalEvent → signals counter."""
        assert isinstance(event, SignalEvent)
        signals_counter.labels(symbol=event.symbol).inc()

    async def _on_bar(self, event: AnyEvent) -> None:
        """BarEvent → bars counter."""
        assert isinstance(event, BarEvent)
        bars_counter.labels(timeframe=event.timeframe).inc()

    async def _on_cb(self, event: AnyEvent) -> None:
        """CircuitBreakerEvent → CB counter."""
        assert isinstance(event, CircuitBreakerEvent)
        cb_triggered_counter.inc()

    async def _on_risk_alert(self, event: AnyEvent) -> None:
        """RiskAlertEvent → risk alerts counter."""
        assert isinstance(event, RiskAlertEvent)
        risk_alerts_counter.labels(level=event.alert_level).inc()

    async def _on_position(self, event: AnyEvent) -> None:
        """PositionUpdateEvent → position gauges."""
        assert isinstance(event, PositionUpdateEvent)
        position_size_gauge.labels(symbol=event.symbol).set(event.size)
