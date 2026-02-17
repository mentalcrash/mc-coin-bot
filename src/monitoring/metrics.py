"""MetricsExporter — Prometheus 메트릭 수출기.

EventBus subscriber로 실시간 메트릭을 수집하고,
prometheus_client HTTP 서버를 통해 /metrics endpoint로 노출합니다.

Layers:
    1. Order Execution — 주문 지연시간/슬리피지/수수료
    2. Position & PnL — 포지션/잔고/레버리지
    3. Exchange API — API 호출/지연시간/WS 상태
    4. Bot Health — uptime/heartbeat/EventBus 상태
    5. Meta — Info/Enum (봇 메타데이터/모드)

Rules Applied:
    - EDA 패턴: EventBus subscribe
    - Prometheus naming: mcbot_ prefix
    - ApiMetricsCallback: Protocol (관심사 분리)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from loguru import logger
from prometheus_client import Counter, Enum, Gauge, Histogram, Info

from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    BarEvent,
    CircuitBreakerEvent,
    EventType,
    FillEvent,
    HeartbeatEvent,
    OrderAckEvent,
    OrderRejectedEvent,
    OrderRequestEvent,
    PositionUpdateEvent,
    RiskAlertEvent,
    SignalEvent,
)

if TYPE_CHECKING:
    from src.core.event_bus import EventBus

# ==========================================================================
# Layer 2: Position & PnL — Gauges (기존)
# ==========================================================================
equity_gauge = Gauge("mcbot_equity_usdt", "Current account equity")
drawdown_gauge = Gauge("mcbot_drawdown_pct", "Current drawdown percentage")
cash_gauge = Gauge("mcbot_cash_usdt", "Available cash")
position_count_gauge = Gauge("mcbot_open_positions", "Number of open positions")
position_size_gauge = Gauge("mcbot_position_size", "Position size", ["symbol"])
uptime_gauge = Gauge("mcbot_uptime_seconds", "Bot uptime in seconds")

# Layer 2: Position & PnL — 신규
position_notional_gauge = Gauge(
    "mcbot_position_notional_usdt", "Position notional value", ["symbol"]
)
unrealized_pnl_gauge = Gauge("mcbot_unrealized_pnl_usdt", "Unrealized PnL", ["symbol"])
realized_profit_counter = Counter(
    "mcbot_realized_profit_usdt_total", "Cumulative realized profit", ["symbol"]
)
realized_loss_counter = Counter(
    "mcbot_realized_loss_usdt_total", "Cumulative realized loss", ["symbol"]
)
aggregate_leverage_gauge = Gauge("mcbot_aggregate_leverage", "Portfolio aggregate leverage ratio")
margin_used_gauge = Gauge("mcbot_margin_used_usdt", "Margin currently in use")

# ==========================================================================
# Layer 1: Order Execution — 기존 Counter
# ==========================================================================
fills_counter = Counter("mcbot_fills", "Total fills executed", ["symbol", "side"])
signals_counter = Counter("mcbot_signals", "Total signals generated", ["symbol"])
bars_counter = Counter("mcbot_bars", "Total bars processed", ["timeframe"])
cb_triggered_counter = Counter("mcbot_circuit_breaker", "Circuit breaker activations")
risk_alerts_counter = Counter("mcbot_risk_alerts", "Risk alerts", ["level"])

# Layer 1: Order Execution — 신규
orders_counter = Counter(
    "mcbot_orders_total",
    "Order counts by status",
    ["symbol", "side", "order_type", "status"],
)
order_latency_histogram = Histogram(
    "mcbot_order_latency_seconds",
    "Order request to fill latency",
    ["symbol"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
slippage_histogram = Histogram(
    "mcbot_slippage_bps",
    "Slippage in basis points",
    ["symbol", "side"],
    buckets=(0, 1, 2, 5, 10, 20, 50, 100),
)
order_rejected_counter = Counter(
    "mcbot_order_rejected_total",
    "Rejected order counts by reason",
    ["symbol", "reason"],
)
fees_counter = Counter("mcbot_fees_usdt_total", "Cumulative fees in USDT", ["symbol"])

# ==========================================================================
# Layer 3: Exchange API
# ==========================================================================
exchange_api_calls_counter = Counter(
    "mcbot_exchange_api_calls_total",
    "Exchange API call counts",
    ["endpoint", "status"],
)
exchange_api_latency_histogram = Histogram(
    "mcbot_exchange_api_latency_seconds",
    "Exchange API response latency",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
exchange_ws_connected_gauge = Gauge(
    "mcbot_exchange_ws_connected",
    "WebSocket connection status (1=connected, 0=disconnected)",
    ["symbol"],
)
exchange_consecutive_failures_gauge = Gauge(
    "mcbot_exchange_consecutive_failures",
    "Consecutive API failure count",
)
ws_reconnects_counter = Counter(
    "mcbot_ws_reconnects_total",
    "WebSocket reconnection count",
    ["symbol"],
)
ws_messages_counter = Counter(
    "mcbot_ws_messages_received_total",
    "WebSocket messages received",
    ["symbol"],
)
ws_last_message_age_gauge = Gauge(
    "mcbot_ws_last_message_age_seconds",
    "Seconds since last WebSocket message",
    ["symbol"],
)

# ==========================================================================
# Layer 4: Bot Health
# ==========================================================================
heartbeat_timestamp_gauge = Gauge("mcbot_heartbeat_timestamp", "Last heartbeat Unix timestamp")
last_bar_age_gauge = Gauge("mcbot_last_bar_age_seconds", "Seconds since last bar", ["symbol"])
errors_counter = Counter(
    "mcbot_errors_total", "Error counts by component", ["component", "error_type"]
)
eventbus_queue_depth_gauge = Gauge("mcbot_eventbus_queue_depth", "EventBus pending event count")
eventbus_events_dropped_counter = Counter(
    "mcbot_eventbus_events_dropped_total", "EventBus dropped events"
)
eventbus_handler_errors_counter = Counter(
    "mcbot_eventbus_handler_errors_total", "EventBus handler errors"
)

# ==========================================================================
# Layer 9: On-chain Data
# ==========================================================================
onchain_fetch_total = Counter(
    "mcbot_onchain_fetch_total",
    "On-chain fetch attempts",
    ["source", "status"],  # status: success | failure | empty
)
onchain_fetch_latency_histogram = Histogram(
    "mcbot_onchain_fetch_latency_seconds",
    "Fetch latency per source",
    ["source"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)
onchain_fetch_rows_gauge = Gauge(
    "mcbot_onchain_fetch_rows",
    "Rows returned by last fetch",
    ["source", "name"],
)
onchain_last_success_gauge = Gauge(
    "mcbot_onchain_last_success_timestamp",
    "Last successful fetch Unix timestamp",
    ["source"],
)
onchain_cache_size_gauge = Gauge(
    "mcbot_onchain_cache_size",
    "Cached on-chain columns per symbol",
    ["symbol"],
)
onchain_cache_refresh_total = Counter(
    "mcbot_onchain_cache_refresh_total",
    "Cache refresh count",
    ["status"],  # success | failure
)

# ==========================================================================
# Meta
# ==========================================================================
bot_info = Info("mcbot", "Bot metadata")
trading_mode_enum = Enum(
    "mcbot_trading_mode",
    "Current trading mode",
    states=["backtest", "paper", "shadow", "live"],
)


# ==========================================================================
# _PendingOrder — 주문 추적용 내부 dataclass
# ==========================================================================
@dataclass
class _PendingOrder:
    """주문 요청→체결 추적용 임시 데이터."""

    symbol: str
    side: str
    order_type: str
    request_time: float  # time.monotonic()
    expected_price: float | None = None  # 슬리피지 기준가
    strategy_name: str = "unknown"


# ==========================================================================
# ApiMetricsCallback — 거래소 API 계측 Protocol
# ==========================================================================
@runtime_checkable
class ApiMetricsCallback(Protocol):
    """거래소 API 호출 메트릭 콜백 Protocol.

    BinanceFuturesClient에 주입하여 관심사 분리.
    """

    def on_api_call(self, endpoint: str, duration: float, status: str) -> None:
        """API 호출 결과 기록.

        Args:
            endpoint: API endpoint 이름 (예: "create_order")
            duration: 호출 소요 시간 (초)
            status: "success" | "retry" | "failure"
        """
        ...


class PrometheusApiCallback:
    """Prometheus 기반 API 메트릭 콜백 구현."""

    def on_api_call(self, endpoint: str, duration: float, status: str) -> None:
        """API 호출 결과를 Prometheus 메트릭으로 기록."""
        exchange_api_calls_counter.labels(endpoint=endpoint, status=status).inc()
        exchange_api_latency_histogram.labels(endpoint=endpoint).observe(duration)


# ==========================================================================
# OnchainMetricsCallback — On-chain 데이터 수집 계측
# ==========================================================================
@runtime_checkable
class OnchainMetricsCallback(Protocol):
    """On-chain 데이터 수집 메트릭 콜백 Protocol.

    route_fetch()에 주입하여 관심사 분리.
    """

    def on_fetch(
        self, source: str, name: str, duration: float, status: str, row_count: int
    ) -> None:
        """Fetch 결과 기록.

        Args:
            source: 데이터 소스 (defillama, coinmetrics, ...)
            name: 데이터 이름 (stablecoin_total, ...)
            duration: 소요 시간 (초)
            status: "success" | "failure" | "empty"
            row_count: 반환된 행 수
        """
        ...


class PrometheusOnchainCallback:
    """Prometheus 기반 On-chain 메트릭 콜백 구현."""

    def on_fetch(
        self, source: str, name: str, duration: float, status: str, row_count: int
    ) -> None:
        """Fetch 결과를 Prometheus 메트릭으로 기록."""
        onchain_fetch_total.labels(source=source, status=status).inc()
        onchain_fetch_latency_histogram.labels(source=source).observe(duration)
        if status == "success" and row_count > 0:
            onchain_fetch_rows_gauge.labels(source=source, name=name).set(row_count)
            onchain_last_success_gauge.labels(source=source).set(time.time())


# ==========================================================================
# WsStatusCallback — WebSocket 상태 콜백
# ==========================================================================
@runtime_checkable
class WsStatusCallback(Protocol):
    """WebSocket 연결 상태 콜백 Protocol."""

    def on_ws_status(self, symbol: str, *, connected: bool) -> None:
        """WS 연결 상태 변경.

        Args:
            symbol: 거래 심볼
            connected: 연결 상태
        """
        ...

    def on_ws_reconnect(self, symbol: str) -> None:
        """WS 재연결 발생.

        Args:
            symbol: 거래 심볼
        """
        ...

    def on_ws_message(self, symbol: str) -> None:
        """WS 메시지 수신.

        Args:
            symbol: 거래 심볼
        """
        ...


class PrometheusWsCallback:
    """Prometheus 기반 WS 상태 콜백 구현 (기본)."""

    def on_ws_status(self, symbol: str, *, connected: bool) -> None:
        """WS 연결 상태를 Prometheus gauge로 기록."""
        exchange_ws_connected_gauge.labels(symbol=symbol).set(1 if connected else 0)

    def on_ws_reconnect(self, symbol: str) -> None:
        """WS 재연결 (no-op — 상세 추적은 PrometheusWsDetailCallback 사용)."""

    def on_ws_message(self, symbol: str) -> None:
        """WS 메시지 수신 (no-op — 상세 추적은 PrometheusWsDetailCallback 사용)."""


class PrometheusWsDetailCallback:
    """기존 WsStatusCallback 호환 + 상세 WS 메트릭 수집.

    on_ws_status: 연결 상태 gauge
    on_ws_message: 메시지 수 counter + 마지막 메시지 시각 기록
    on_ws_reconnect: 재연결 counter
    update_message_ages: 주기적 호출로 last_message_age gauge 갱신
    """

    def __init__(self) -> None:
        self._last_message_time: dict[str, float] = {}

    def on_ws_status(self, symbol: str, *, connected: bool) -> None:
        """WS 연결 상태를 Prometheus gauge로 기록."""
        exchange_ws_connected_gauge.labels(symbol=symbol).set(1 if connected else 0)

    def on_ws_message(self, symbol: str) -> None:
        """WS 메시지 수신 → counter 증가 + 시각 기록."""
        ws_messages_counter.labels(symbol=symbol).inc()
        self._last_message_time[symbol] = time.monotonic()

    def on_ws_reconnect(self, symbol: str) -> None:
        """WS 재연결 → counter 증가."""
        ws_reconnects_counter.labels(symbol=symbol).inc()

    def update_message_ages(self) -> None:
        """심볼별 마지막 메시지 경과 시간을 gauge로 갱신."""
        now = time.monotonic()
        for sym, last in self._last_message_time.items():
            ws_last_message_age_gauge.labels(symbol=sym).set(now - last)


# ==========================================================================
# Rejection reason 분류
# ==========================================================================
_REJECTION_REASON_MAP: dict[str, str] = {
    "leverage": "leverage_exceeded",
    "max_positions": "max_positions",
    "order_size": "order_size_exceeded",
    "circuit_breaker": "circuit_breaker",
    "circuit breaker": "circuit_breaker",
}


def _categorize_reason(reason: str) -> str:
    """거부 사유 문자열 → 표준 카테고리.

    Args:
        reason: OrderRejectedEvent.reason 원본 문자열

    Returns:
        표준화된 거부 사유 (매칭 실패 시 "other")
    """
    lower = reason.lower()
    for keyword, category in _REJECTION_REASON_MAP.items():
        if keyword in lower:
            return category
    return "other"


def _calculate_slippage_bps(expected_price: float, fill_price: float) -> float:
    """기대가 vs 체결가 차이를 basis points로 계산.

    Args:
        expected_price: 기준가 (bar close 또는 limit price)
        fill_price: 실제 체결가

    Returns:
        슬리피지 (basis points, 항상 양수). 기대가 0이면 0.0
    """
    if expected_price <= 0:
        return 0.0
    return abs(fill_price - expected_price) / expected_price * 10000


# ==========================================================================
# MetricsExporter
# ==========================================================================
@dataclass
class _EventBusSnapshot:
    """EventBus 메트릭 이전 스냅샷 (delta 계산용)."""

    events_dropped: int = 0
    handler_errors: int = 0


# ==========================================================================
# Per-strategy metrics
# ==========================================================================
# ==========================================================================
# Per-strategy anomaly detection
# ==========================================================================
distribution_ks_statistic_gauge = Gauge(
    "mcbot_distribution_ks_statistic", "Distribution drift KS statistic", ["strategy"]
)
distribution_p_value_gauge = Gauge(
    "mcbot_distribution_p_value", "Distribution drift p-value", ["strategy"]
)
ransac_slope_gauge = Gauge("mcbot_ransac_slope", "RANSAC estimated slope", ["strategy"])
ransac_conformal_lower_gauge = Gauge(
    "mcbot_ransac_conformal_lower", "RANSAC conformal lower bound", ["strategy"]
)
ransac_decay_detected_gauge = Gauge(
    "mcbot_ransac_decay_detected", "RANSAC structural decay detected (0 or 1)", ["strategy"]
)

strategy_pnl_gauge = Gauge("mcbot_strategy_pnl_usdt", "Strategy PnL", ["strategy"])
strategy_signals_counter = Counter(
    "mcbot_strategy_signals_total", "Signals by strategy", ["strategy", "side"]
)
strategy_fills_counter = Counter(
    "mcbot_strategy_fills_total", "Fills by strategy", ["strategy", "side"]
)
strategy_fees_counter = Counter("mcbot_strategy_fees_usdt_total", "Fees by strategy", ["strategy"])


def _extract_strategy(client_order_id: str) -> str:
    """client_order_id에서 전략명 추출.

    Format: ``{strategy}-{symbol_slug}-{counter}`` (예: ``ctrend-BTCUSDT-42``).
    ``rsplit("-", 2)`` 로 마지막 2개 세그먼트를 분리하고 나머지를 전략명으로 사용합니다.

    Args:
        client_order_id: 주문 멱등성 키

    Returns:
        전략명 (파싱 실패 시 "unknown")
    """
    parts = client_order_id.rsplit("-", 2)
    if len(parts) >= 3:  # noqa: PLR2004
        return parts[0]
    return "unknown"


# ==========================================================================
# Execution quality alert 임계값
# ==========================================================================
_SLIPPAGE_WARN_BPS = 15.0
_SLIPPAGE_CRITICAL_BPS = 30.0
_LATENCY_WARN_SECONDS = 5.0
_LATENCY_CRITICAL_SECONDS = 10.0


class MetricsExporter:
    """Prometheus 메트릭 수출기 — EventBus subscriber.

    Args:
        port: Prometheus HTTP 서버 포트 (0=비활성)
    """

    def __init__(self, port: int = 8000) -> None:
        from src.monitoring.anomaly.execution_quality import ExecutionAnomalyDetector

        self._port = port
        self._start_time = time.monotonic()
        self._pending_orders: dict[str, _PendingOrder] = {}
        self._last_bar_close: dict[str, float] = {}
        self._last_bar_time: dict[str, float] = {}  # symbol → monotonic timestamp
        self._eventbus_snapshot = _EventBusSnapshot()
        self._bus: EventBus | None = None
        self._anomaly_detector = ExecutionAnomalyDetector()

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록.

        Args:
            bus: EventBus 인스턴스
        """
        self._bus = bus
        bus.subscribe(EventType.BALANCE_UPDATE, self._on_balance)
        bus.subscribe(EventType.FILL, self._on_fill)
        bus.subscribe(EventType.SIGNAL, self._on_signal)
        bus.subscribe(EventType.BAR, self._on_bar)
        bus.subscribe(EventType.CIRCUIT_BREAKER, self._on_cb)
        bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert)
        bus.subscribe(EventType.POSITION_UPDATE, self._on_position)
        bus.subscribe(EventType.ORDER_REQUEST, self._on_order_request)
        bus.subscribe(EventType.ORDER_ACK, self._on_order_ack)
        bus.subscribe(EventType.ORDER_REJECTED, self._on_order_rejected)
        bus.subscribe(EventType.HEARTBEAT, self._on_heartbeat)
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

    def update_eventbus_metrics(self, bus: EventBus) -> None:
        """EventBus 내부 메트릭을 Prometheus로 export (delta 방식).

        Args:
            bus: EventBus 인스턴스
        """
        eventbus_queue_depth_gauge.set(bus.queue_size)

        # Delta 방식: 이전 스냅샷 대비 증가분만 counter에 반영
        current_dropped = bus.metrics.events_dropped
        delta_dropped = current_dropped - self._eventbus_snapshot.events_dropped
        if delta_dropped > 0:
            eventbus_events_dropped_counter.inc(delta_dropped)
        self._eventbus_snapshot.events_dropped = current_dropped

        current_errors = bus.metrics.handler_errors
        delta_errors = current_errors - self._eventbus_snapshot.handler_errors
        if delta_errors > 0:
            eventbus_handler_errors_counter.inc(delta_errors)
        self._eventbus_snapshot.handler_errors = current_errors

    def update_exchange_health(self, consecutive_failures: int) -> None:
        """거래소 API 연속 실패 횟수 갱신.

        Args:
            consecutive_failures: 현재 연속 실패 횟수
        """
        exchange_consecutive_failures_gauge.set(consecutive_failures)

    def update_bar_ages(self) -> None:
        """각 심볼의 마지막 bar 이후 경과 시간(초)을 gauge로 갱신."""
        now = time.monotonic()
        for symbol, last_time in self._last_bar_time.items():
            age = now - last_time
            last_bar_age_gauge.labels(symbol=symbol).set(age)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    async def _on_balance(self, event: AnyEvent) -> None:
        """BalanceUpdateEvent → equity/cash/margin gauges."""
        assert isinstance(event, BalanceUpdateEvent)
        equity_gauge.set(event.total_equity)
        cash_gauge.set(event.available_cash)
        margin_used_gauge.set(event.total_margin_used)

    async def _on_fill(self, event: AnyEvent) -> None:
        """FillEvent → fills counter + latency + slippage + fees + execution quality alerts."""
        assert isinstance(event, FillEvent)
        fills_counter.labels(symbol=event.symbol, side=event.side).inc()

        # 수수료 추적
        if event.fee > 0:
            fees_counter.labels(symbol=event.symbol).inc(event.fee)

        # pending order 매칭 → latency + slippage
        pending = self._pending_orders.pop(event.client_order_id, None)
        if pending is not None:
            # 주문 지연시간
            latency = time.monotonic() - pending.request_time
            order_latency_histogram.labels(symbol=event.symbol).observe(latency)

            # Latency alert
            await self._check_latency_alert(latency, event.symbol)

            # orders_total (filled)
            orders_counter.labels(
                symbol=pending.symbol,
                side=pending.side,
                order_type=pending.order_type,
                status="filled",
            ).inc()

            # 슬리피지 계산
            expected = pending.expected_price
            if expected is not None and expected > 0:
                bps = _calculate_slippage_bps(expected, event.fill_price)
                slippage_histogram.labels(symbol=event.symbol, side=event.side).observe(bps)

                # Slippage alert
                await self._check_slippage_alert(bps, event.symbol)

            # Per-strategy metrics
            strategy_fills_counter.labels(strategy=pending.strategy_name, side=event.side).inc()
            if event.fee > 0:
                strategy_fees_counter.labels(strategy=pending.strategy_name).inc(event.fee)

            # Execution anomaly detection
            bps_for_anomaly = (
                _calculate_slippage_bps(expected, event.fill_price)
                if expected and expected > 0
                else 0.0
            )
            anomalies = self._anomaly_detector.on_fill(latency, bps_for_anomaly)
            for anomaly in anomalies:
                await self._publish_execution_alert(anomaly.severity, anomaly.message)

    async def _on_signal(self, event: AnyEvent) -> None:
        """SignalEvent → signals counter + strategy signals counter."""
        assert isinstance(event, SignalEvent)
        signals_counter.labels(symbol=event.symbol).inc()
        strategy_signals_counter.labels(
            strategy=event.strategy_name, side=event.direction.name
        ).inc()

    async def _on_bar(self, event: AnyEvent) -> None:
        """BarEvent → bars counter + last_bar_close + bar age 갱신."""
        assert isinstance(event, BarEvent)
        bars_counter.labels(timeframe=event.timeframe).inc()
        self._last_bar_close[event.symbol] = event.close
        self._last_bar_time[event.symbol] = time.monotonic()
        last_bar_age_gauge.labels(symbol=event.symbol).set(0)

    async def _on_cb(self, event: AnyEvent) -> None:
        """CircuitBreakerEvent → CB counter."""
        assert isinstance(event, CircuitBreakerEvent)
        cb_triggered_counter.inc()

    async def _on_risk_alert(self, event: AnyEvent) -> None:
        """RiskAlertEvent → risk alerts counter."""
        assert isinstance(event, RiskAlertEvent)
        risk_alerts_counter.labels(level=event.alert_level).inc()

    async def _on_position(self, event: AnyEvent) -> None:
        """PositionUpdateEvent → position gauges + notional + unrealized PnL."""
        assert isinstance(event, PositionUpdateEvent)
        position_size_gauge.labels(symbol=event.symbol).set(event.size)
        notional = abs(event.size * event.avg_entry_price)
        position_notional_gauge.labels(symbol=event.symbol).set(notional)
        unrealized_pnl_gauge.labels(symbol=event.symbol).set(event.unrealized_pnl)

        # 실현 손익: profit/loss 분리 (Counter는 단조 증가만 가능)
        if event.realized_pnl > 0:
            realized_profit_counter.labels(symbol=event.symbol).inc(event.realized_pnl)
        elif event.realized_pnl < 0:
            realized_loss_counter.labels(symbol=event.symbol).inc(abs(event.realized_pnl))

    async def _on_order_request(self, event: AnyEvent) -> None:
        """OrderRequestEvent → pending 등록 + 기준가 저장."""
        assert isinstance(event, OrderRequestEvent)

        # 기준가 결정: 지정가 주문은 order.price, 시장가는 last bar close
        if event.price is not None:
            expected_price = event.price
        else:
            expected_price = self._last_bar_close.get(event.symbol)

        self._pending_orders[event.client_order_id] = _PendingOrder(
            symbol=event.symbol,
            side=event.side,
            order_type=event.order_type,
            request_time=time.monotonic(),
            expected_price=expected_price,
            strategy_name=_extract_strategy(event.client_order_id),
        )
        self._anomaly_detector.on_order_request()

    async def _on_order_ack(self, event: AnyEvent) -> None:
        """OrderAckEvent → orders_total (ack) 증가."""
        assert isinstance(event, OrderAckEvent)
        # ACK는 접수 확인일 뿐 — pending은 유지 (fill 시 제거)
        pending = self._pending_orders.get(event.client_order_id)
        if pending is not None:
            orders_counter.labels(
                symbol=pending.symbol,
                side=pending.side,
                order_type=pending.order_type,
                status="ack",
            ).inc()

    async def _on_order_rejected(self, event: AnyEvent) -> None:
        """OrderRejectedEvent → rejected counter + orders_total (rejected)."""
        assert isinstance(event, OrderRejectedEvent)
        reason_category = _categorize_reason(event.reason)
        order_rejected_counter.labels(symbol=event.symbol, reason=reason_category).inc()

        # Execution anomaly detection — rejection tracking
        rejection_anomalies = self._anomaly_detector.on_rejection()
        for anomaly in rejection_anomalies:
            await self._publish_execution_alert(anomaly.severity, anomaly.message)

        # pending 있으면 orders_total도 기록 후 제거
        pending = self._pending_orders.pop(event.client_order_id, None)
        if pending is not None:
            orders_counter.labels(
                symbol=pending.symbol,
                side=pending.side,
                order_type=pending.order_type,
                status="rejected",
            ).inc()

    async def _on_heartbeat(self, event: AnyEvent) -> None:
        """HeartbeatEvent → heartbeat timestamp gauge."""
        assert isinstance(event, HeartbeatEvent)
        heartbeat_timestamp_gauge.set(event.timestamp.timestamp())

    def check_execution_health(self) -> object | None:
        """Fill rate 주기적 체크용 (external caller).

        Returns:
            ExecutionAnomaly 또는 None (정상/데이터 부족)
        """
        return self._anomaly_detector.check_fill_rate()

    # ------------------------------------------------------------------
    # Execution quality alerts
    # ------------------------------------------------------------------
    async def _check_slippage_alert(self, bps: float, symbol: str) -> None:
        """슬리피지 임계값 초과 시 RiskAlertEvent 발행."""
        if bps > _SLIPPAGE_CRITICAL_BPS:
            await self._publish_execution_alert(
                "CRITICAL", f"High slippage {bps:.1f}bps on {symbol}"
            )
        elif bps > _SLIPPAGE_WARN_BPS:
            await self._publish_execution_alert(
                "WARNING", f"Elevated slippage {bps:.1f}bps on {symbol}"
            )

    async def _check_latency_alert(self, latency: float, symbol: str) -> None:
        """체결 지연시간 임계값 초과 시 RiskAlertEvent 발행."""
        if latency > _LATENCY_CRITICAL_SECONDS:
            await self._publish_execution_alert(
                "CRITICAL", f"High fill latency {latency:.1f}s on {symbol}"
            )
        elif latency > _LATENCY_WARN_SECONDS:
            await self._publish_execution_alert(
                "WARNING", f"Elevated fill latency {latency:.1f}s on {symbol}"
            )

    async def _publish_execution_alert(
        self, level: Literal["WARNING", "CRITICAL"], message: str
    ) -> None:
        """RiskAlertEvent를 EventBus에 발행.

        Args:
            level: "WARNING" or "CRITICAL"
            message: 알림 메시지
        """
        if self._bus is None:
            return
        alert = RiskAlertEvent(
            alert_level=level,
            message=message,
            source="MetricsExporter:execution_quality",
        )
        await self._bus.publish(alert)
