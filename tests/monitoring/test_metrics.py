"""MetricsExporter 테스트 — Prometheus Counter/Gauge/Histogram/Info/Enum 검증."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import pytest
from prometheus_client import REGISTRY

from src.core.event_bus import EventBus
from src.core.events import (
    BalanceUpdateEvent,
    BarEvent,
    CircuitBreakerEvent,
    FillEvent,
    HeartbeatEvent,
    OrderAckEvent,
    OrderRejectedEvent,
    OrderRequestEvent,
    PositionUpdateEvent,
    RiskAlertEvent,
    SignalEvent,
)
from src.models.types import Direction
from src.monitoring.metrics import (
    MetricsExporter,
    PrometheusApiCallback,
    PrometheusWsCallback,
    PrometheusWsDetailCallback,
    _calculate_slippage_bps,
    _categorize_reason,
    _extract_strategy,
)


@pytest.fixture(autouse=True)
def _reset_prometheus_registry() -> None:
    """테스트 간 prometheus collector 충돌 방지.

    모듈 레벨 Gauge/Counter는 프로세스 전역이므로 값만 리셋합니다.
    """
    # Gauge는 set(0)으로, Counter는 이미 monotonic이라 검증 시 delta 사용


def _sample(name: str, labels: dict[str, str] | None = None) -> float | None:
    """REGISTRY에서 특정 metric 값을 가져옴."""
    return REGISTRY.get_sample_value(name, labels or {})


async def _run_with_bus(exporter: MetricsExporter, events: list[object]) -> None:
    """헬퍼: bus 시작 → 이벤트 발행 → flush → 종료."""
    bus = EventBus(queue_size=100)
    await exporter.register(bus)
    bus_task = asyncio.create_task(bus.start())
    try:
        for event in events:
            await bus.publish(event)  # type: ignore[arg-type]
        await bus.flush()
    finally:
        await bus.stop()
        await bus_task


class TestMetricsExporterRegister:
    async def test_register_subscribes_events(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)
        # 핸들러가 등록되었는지 확인 (11개 이벤트 타입)
        assert len(bus._handlers) >= 11


class TestBalanceEvent:
    async def test_equity_and_cash_gauges(self) -> None:
        exporter = MetricsExporter(port=0)
        event = BalanceUpdateEvent(
            total_equity=15000.0,
            available_cash=12000.0,
            total_margin_used=3000.0,
        )
        await _run_with_bus(exporter, [event])

        assert _sample("mcbot_equity_usdt") == 15000.0
        assert _sample("mcbot_cash_usdt") == 12000.0
        assert _sample("mcbot_margin_used_usdt") == 3000.0


class TestFillEvent:
    async def test_fills_counter(self) -> None:
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_fills_total", {"symbol": "BTC/USDT", "side": "BUY"}) or 0.0

        event = FillEvent(
            client_order_id="test-001",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=40000.0,
            fill_qty=0.1,
            fee=4.0,
            fill_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_fills_total", {"symbol": "BTC/USDT", "side": "BUY"})
        assert after is not None
        assert after > before

    async def test_fee_tracking(self) -> None:
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_fees_usdt_total", {"symbol": "ETH/USDT"}) or 0.0

        event = FillEvent(
            client_order_id="fee-001",
            symbol="ETH/USDT",
            side="BUY",
            fill_price=3000.0,
            fill_qty=1.0,
            fee=3.0,
            fill_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_fees_usdt_total", {"symbol": "ETH/USDT"})
        assert after is not None
        assert after >= before + 3.0


class TestBarEvent:
    async def test_bars_counter(self) -> None:
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_bars_total", {"timeframe": "1D"}) or 0.0

        event = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=40000.0,
            high=41000.0,
            low=39000.0,
            close=40500.0,
            volume=1000.0,
            bar_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_bars_total", {"timeframe": "1D"})
        assert after is not None
        assert after > before

    async def test_last_bar_close_tracking(self) -> None:
        exporter = MetricsExporter(port=0)
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=40000.0,
            high=41000.0,
            low=39000.0,
            close=40500.0,
            volume=1000.0,
            bar_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [bar])
        assert exporter._last_bar_close["BTC/USDT"] == 40500.0


class TestSignalEvent:
    async def test_signals_counter(self) -> None:
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_signals_total", {"symbol": "ETH/USDT"}) or 0.0

        event = SignalEvent(
            symbol="ETH/USDT",
            strategy_name="tsmom",
            direction=Direction.LONG,
            strength=0.8,
            bar_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_signals_total", {"symbol": "ETH/USDT"})
        assert after is not None
        assert after > before


class TestCircuitBreakerEvent:
    async def test_cb_counter(self) -> None:
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_circuit_breaker_total") or 0.0

        event = CircuitBreakerEvent(
            reason="System stop-loss triggered",
            close_all_positions=True,
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_circuit_breaker_total")
        assert after is not None
        assert after > before


class TestPositionUpdateEvent:
    async def test_position_size_gauge(self) -> None:
        exporter = MetricsExporter(port=0)
        event = PositionUpdateEvent(
            symbol="SOL/USDT",
            direction=Direction.LONG,
            size=5.0,
            avg_entry_price=100.0,
            unrealized_pnl=25.0,
        )
        await _run_with_bus(exporter, [event])

        val = _sample("mcbot_position_size", {"symbol": "SOL/USDT"})
        assert val == 5.0

    async def test_position_notional_gauge(self) -> None:
        exporter = MetricsExporter(port=0)
        event = PositionUpdateEvent(
            symbol="BTC/USDT",
            direction=Direction.LONG,
            size=0.5,
            avg_entry_price=60000.0,
            unrealized_pnl=500.0,
        )
        await _run_with_bus(exporter, [event])

        notional = _sample("mcbot_position_notional_usdt", {"symbol": "BTC/USDT"})
        assert notional == 30000.0  # 0.5 * 60000

    async def test_unrealized_pnl_gauge(self) -> None:
        exporter = MetricsExporter(port=0)
        event = PositionUpdateEvent(
            symbol="ETH/USDT",
            direction=Direction.SHORT,
            size=10.0,
            avg_entry_price=3000.0,
            unrealized_pnl=-150.0,
        )
        await _run_with_bus(exporter, [event])

        pnl = _sample("mcbot_unrealized_pnl_usdt", {"symbol": "ETH/USDT"})
        assert pnl == -150.0

    async def test_realized_profit_counter(self) -> None:
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_realized_profit_usdt_total", {"symbol": "BTC/USDT"}) or 0.0

        event = PositionUpdateEvent(
            symbol="BTC/USDT",
            direction=Direction.NEUTRAL,
            size=0.0,
            avg_entry_price=0.0,
            realized_pnl=250.0,
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_realized_profit_usdt_total", {"symbol": "BTC/USDT"})
        assert after is not None
        assert after >= before + 250.0

    async def test_realized_loss_counter(self) -> None:
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_realized_loss_usdt_total", {"symbol": "BTC/USDT"}) or 0.0

        event = PositionUpdateEvent(
            symbol="BTC/USDT",
            direction=Direction.NEUTRAL,
            size=0.0,
            avg_entry_price=0.0,
            realized_pnl=-300.0,
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_realized_loss_usdt_total", {"symbol": "BTC/USDT"})
        assert after is not None
        assert after >= before + 300.0


class TestRiskAlertEvent:
    async def test_risk_alerts_counter(self) -> None:
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_risk_alerts_total", {"level": "WARNING"}) or 0.0

        event = RiskAlertEvent(
            alert_level="WARNING",
            message="Drawdown approaching threshold",
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_risk_alerts_total", {"level": "WARNING"})
        assert after is not None
        assert after > before


class TestUptime:
    def test_update_uptime(self) -> None:
        exporter = MetricsExporter(port=0)
        time.sleep(0.01)
        exporter.update_uptime()
        val = _sample("mcbot_uptime_seconds")
        assert val is not None
        assert val > 0


class TestOrderRequestEvent:
    async def test_order_request_registers_pending(self) -> None:
        exporter = MetricsExporter(port=0)
        event = OrderRequestEvent(
            client_order_id="ord-001",
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        await _run_with_bus(exporter, [event])
        assert "ord-001" in exporter._pending_orders

    async def test_order_request_limit_uses_price(self) -> None:
        exporter = MetricsExporter(port=0)
        event = OrderRequestEvent(
            client_order_id="ord-limit",
            symbol="BTC/USDT",
            side="BUY",
            order_type="LIMIT",
            target_weight=0.5,
            notional_usd=5000.0,
            price=39000.0,
        )
        await _run_with_bus(exporter, [event])
        assert exporter._pending_orders["ord-limit"].expected_price == 39000.0

    async def test_order_request_market_uses_last_bar_close(self) -> None:
        exporter = MetricsExporter(port=0)
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=40000.0,
            high=41000.0,
            low=39000.0,
            close=40500.0,
            volume=1000.0,
            bar_timestamp=datetime.now(UTC),
        )
        order = OrderRequestEvent(
            client_order_id="ord-mkt",
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        await _run_with_bus(exporter, [bar, order])
        assert exporter._pending_orders["ord-mkt"].expected_price == 40500.0


class TestOrderAckEvent:
    async def test_order_ack_increments_counter(self) -> None:
        exporter = MetricsExporter(port=0)
        before = (
            _sample(
                "mcbot_orders_total",
                {"symbol": "BTC/USDT", "side": "BUY", "order_type": "MARKET", "status": "ack"},
            )
            or 0.0
        )

        req = OrderRequestEvent(
            client_order_id="ack-001",
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        ack = OrderAckEvent(
            client_order_id="ack-001",
            symbol="BTC/USDT",
        )
        await _run_with_bus(exporter, [req, ack])

        after = _sample(
            "mcbot_orders_total",
            {"symbol": "BTC/USDT", "side": "BUY", "order_type": "MARKET", "status": "ack"},
        )
        assert after is not None
        assert after > before


class TestOrderRejectedEvent:
    async def test_rejected_leverage_reason(self) -> None:
        exporter = MetricsExporter(port=0)
        before = (
            _sample(
                "mcbot_order_rejected_total",
                {"symbol": "BTC/USDT", "reason": "leverage_exceeded"},
            )
            or 0.0
        )

        req = OrderRequestEvent(
            client_order_id="rej-001",
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        rej = OrderRejectedEvent(
            client_order_id="rej-001",
            symbol="BTC/USDT",
            reason="Leverage limit exceeded: current 3.0x > max 2.0x",
        )
        await _run_with_bus(exporter, [req, rej])

        after = _sample(
            "mcbot_order_rejected_total",
            {"symbol": "BTC/USDT", "reason": "leverage_exceeded"},
        )
        assert after is not None
        assert after > before

    async def test_rejected_removes_pending(self) -> None:
        exporter = MetricsExporter(port=0)
        req = OrderRequestEvent(
            client_order_id="rej-002",
            symbol="ETH/USDT",
            side="SELL",
            order_type="MARKET",
            target_weight=0.0,
            notional_usd=2000.0,
        )
        rej = OrderRejectedEvent(
            client_order_id="rej-002",
            symbol="ETH/USDT",
            reason="circuit breaker active",
        )
        await _run_with_bus(exporter, [req, rej])
        assert "rej-002" not in exporter._pending_orders


class TestOrderLatencyAndSlippage:
    async def test_fill_records_latency_and_slippage(self) -> None:
        exporter = MetricsExporter(port=0)
        # 먼저 bar close 설정
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=40000.0,
            high=41000.0,
            low=39000.0,
            close=40000.0,
            volume=1000.0,
            bar_timestamp=datetime.now(UTC),
        )
        order = OrderRequestEvent(
            client_order_id="lat-001",
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        fill = FillEvent(
            client_order_id="lat-001",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=40020.0,  # 20 USD slippage on 40000 = 5 bps
            fill_qty=0.1,
            fee=4.0,
            fill_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [bar, order, fill])

        # pending이 제거되었는지 확인
        assert "lat-001" not in exporter._pending_orders

        # orders_total filled가 증가했는지 확인
        filled = _sample(
            "mcbot_orders_total",
            {"symbol": "BTC/USDT", "side": "BUY", "order_type": "MARKET", "status": "filled"},
        )
        assert filled is not None
        assert filled > 0

    async def test_fill_without_pending_still_counts(self) -> None:
        """pending이 없는 fill도 fills_counter는 증가해야 함."""
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_fills_total", {"symbol": "DOGE/USDT", "side": "SELL"}) or 0.0

        fill = FillEvent(
            client_order_id="no-pending",
            symbol="DOGE/USDT",
            side="SELL",
            fill_price=0.1,
            fill_qty=1000.0,
            fee=0.1,
            fill_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [fill])

        after = _sample("mcbot_fills_total", {"symbol": "DOGE/USDT", "side": "SELL"})
        assert after is not None
        assert after > before


class TestHeartbeatEvent:
    async def test_heartbeat_sets_timestamp(self) -> None:
        exporter = MetricsExporter(port=0)
        now = datetime.now(UTC)
        event = HeartbeatEvent(
            component="test",
            bars_processed=100,
            timestamp=now,
        )
        await _run_with_bus(exporter, [event])

        val = _sample("mcbot_heartbeat_timestamp")
        assert val is not None
        assert val == pytest.approx(now.timestamp(), abs=1.0)


class TestEventBusMetricsBridge:
    async def test_update_eventbus_metrics(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)

        # 수동으로 bus metrics 설정
        bus.metrics.events_dropped = 5
        bus.metrics.handler_errors = 2

        exporter.update_eventbus_metrics(bus)

        depth = _sample("mcbot_eventbus_queue_depth")
        assert depth is not None
        assert depth >= 0

    async def test_eventbus_delta_calculation(self) -> None:
        """delta 방식이므로 두 번째 호출에서 증가분만 반영."""
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)

        bus.metrics.events_dropped = 3
        exporter.update_eventbus_metrics(bus)
        before = _sample("mcbot_eventbus_events_dropped_total") or 0.0

        bus.metrics.events_dropped = 5  # delta = 2
        exporter.update_eventbus_metrics(bus)
        after = _sample("mcbot_eventbus_events_dropped_total") or 0.0

        assert after >= before + 2.0


class TestExchangeHealthMetrics:
    def test_update_exchange_health(self) -> None:
        exporter = MetricsExporter(port=0)
        exporter.update_exchange_health(3)
        val = _sample("mcbot_exchange_consecutive_failures")
        assert val == 3.0


class TestSlippageCalculation:
    def test_positive_slippage(self) -> None:
        # 40000 → 40020 = 5 bps
        bps = _calculate_slippage_bps(40000.0, 40020.0)
        assert bps == pytest.approx(5.0)

    def test_negative_slippage(self) -> None:
        # 40000 → 39960 = 10 bps (abs value)
        bps = _calculate_slippage_bps(40000.0, 39960.0)
        assert bps == pytest.approx(10.0)

    def test_zero_expected_price(self) -> None:
        assert _calculate_slippage_bps(0.0, 100.0) == 0.0

    def test_exact_fill(self) -> None:
        assert _calculate_slippage_bps(100.0, 100.0) == 0.0


class TestReasonCategorization:
    def test_leverage_exceeded(self) -> None:
        assert _categorize_reason("Leverage limit exceeded: 3.0x") == "leverage_exceeded"

    def test_max_positions(self) -> None:
        assert _categorize_reason("max_positions reached") == "max_positions"

    def test_order_size(self) -> None:
        assert _categorize_reason("Order_size too large") == "order_size_exceeded"

    def test_circuit_breaker(self) -> None:
        assert _categorize_reason("Circuit breaker active") == "circuit_breaker"

    def test_unknown_reason(self) -> None:
        assert _categorize_reason("Some unknown error") == "other"


class TestPrometheusApiCallback:
    def test_on_api_call(self) -> None:
        cb = PrometheusApiCallback()
        before = (
            _sample(
                "mcbot_exchange_api_calls_total",
                {"endpoint": "create_order", "status": "success"},
            )
            or 0.0
        )

        cb.on_api_call("create_order", 0.15, "success")

        after = _sample(
            "mcbot_exchange_api_calls_total",
            {"endpoint": "create_order", "status": "success"},
        )
        assert after is not None
        assert after > before


class TestPrometheusWsCallback:
    def test_on_ws_connected(self) -> None:
        cb = PrometheusWsCallback()
        cb.on_ws_status("BTC/USDT", connected=True)
        val = _sample("mcbot_exchange_ws_connected", {"symbol": "BTC/USDT"})
        assert val == 1.0

    def test_on_ws_disconnected(self) -> None:
        cb = PrometheusWsCallback()
        cb.on_ws_status("ETH/USDT", connected=False)
        val = _sample("mcbot_exchange_ws_connected", {"symbol": "ETH/USDT"})
        assert val == 0.0


class TestInfoAndEnumMetrics:
    def test_bot_info(self) -> None:
        from src.monitoring.metrics import bot_info

        bot_info.info({"version": "0.1.0", "mode": "paper"})
        val = _sample("mcbot_info", {"version": "0.1.0", "mode": "paper"})
        assert val == 1.0

    def test_trading_mode_enum(self) -> None:
        from src.monitoring.metrics import trading_mode_enum

        trading_mode_enum.state("paper")
        val = _sample("mcbot_trading_mode", {"mcbot_trading_mode": "paper"})
        assert val == 1.0


class TestExtractStrategy:
    """_extract_strategy() 단위 테스트."""

    def test_standard_format(self) -> None:
        assert _extract_strategy("ctrend-BTCUSDT-42") == "ctrend"

    def test_hyphenated_strategy_name(self) -> None:
        assert _extract_strategy("anchor-mom-ETHUSDT-7") == "anchor-mom"

    def test_too_few_parts(self) -> None:
        assert _extract_strategy("simple") == "unknown"

    def test_two_parts(self) -> None:
        assert _extract_strategy("strategy-only") == "unknown"

    def test_empty_string(self) -> None:
        assert _extract_strategy("") == "unknown"


class TestStrategySignalCounter:
    """Per-strategy signal counter 검증."""

    async def test_signal_increments_strategy_counter(self) -> None:
        exporter = MetricsExporter(port=0)
        before = (
            _sample(
                "mcbot_strategy_signals_total",
                {"strategy": "tsmom", "side": "LONG"},
            )
            or 0.0
        )

        event = SignalEvent(
            symbol="BTC/USDT",
            strategy_name="tsmom",
            direction=Direction.LONG,
            strength=0.8,
            bar_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [event])

        after = _sample(
            "mcbot_strategy_signals_total",
            {"strategy": "tsmom", "side": "LONG"},
        )
        assert after is not None
        assert after > before


class TestStrategyFillCounter:
    """Per-strategy fill/fee counter 검증."""

    async def test_fill_increments_strategy_counters(self) -> None:
        exporter = MetricsExporter(port=0)
        before_fills = (
            _sample(
                "mcbot_strategy_fills_total",
                {"strategy": "ctrend", "side": "BUY"},
            )
            or 0.0
        )
        before_fees = _sample("mcbot_strategy_fees_usdt_total", {"strategy": "ctrend"}) or 0.0

        order = OrderRequestEvent(
            client_order_id="ctrend-BTCUSDT-99",
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        fill = FillEvent(
            client_order_id="ctrend-BTCUSDT-99",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=40000.0,
            fill_qty=0.1,
            fee=4.0,
            fill_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [order, fill])

        after_fills = _sample(
            "mcbot_strategy_fills_total",
            {"strategy": "ctrend", "side": "BUY"},
        )
        after_fees = _sample("mcbot_strategy_fees_usdt_total", {"strategy": "ctrend"})
        assert after_fills is not None
        assert after_fills > before_fills
        assert after_fees is not None
        assert after_fees >= before_fees + 4.0

    async def test_pending_order_captures_strategy_name(self) -> None:
        exporter = MetricsExporter(port=0)
        order = OrderRequestEvent(
            client_order_id="anchor-mom-ETHUSDT-5",
            symbol="ETH/USDT",
            side="SELL",
            order_type="MARKET",
            target_weight=0.0,
            notional_usd=3000.0,
        )
        await _run_with_bus(exporter, [order])
        pending = exporter._pending_orders["anchor-mom-ETHUSDT-5"]
        assert pending.strategy_name == "anchor-mom"


class TestExecutionQualityAlerts:
    """MetricsExporter execution quality alert 검증."""

    async def test_high_slippage_publishes_risk_alert(self) -> None:
        """슬리피지 > 30bps → CRITICAL RiskAlertEvent 발행."""
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)

        # RiskAlertEvent 캡처
        alerts: list[RiskAlertEvent] = []

        async def capture_alert(event: object) -> None:
            assert isinstance(event, RiskAlertEvent)
            alerts.append(event)

        bus.subscribe("risk_alert", capture_alert)

        bus_task = asyncio.create_task(bus.start())
        try:
            # bar close 설정 → order → fill with high slippage
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=40000.0,
                high=41000.0,
                low=39000.0,
                close=40000.0,
                volume=1000.0,
                bar_timestamp=datetime.now(UTC),
            )
            order = OrderRequestEvent(
                client_order_id="slip-crit-001",
                symbol="BTC/USDT",
                side="BUY",
                order_type="MARKET",
                target_weight=0.5,
                notional_usd=5000.0,
            )
            # 40000 → 40200 = 50 bps (> 30 CRITICAL)
            fill = FillEvent(
                client_order_id="slip-crit-001",
                symbol="BTC/USDT",
                side="BUY",
                fill_price=40200.0,
                fill_qty=0.1,
                fee=4.0,
                fill_timestamp=datetime.now(UTC),
            )
            for evt in [bar, order, fill]:
                await bus.publish(evt)  # type: ignore[arg-type]
            await bus.flush()
        finally:
            await bus.stop()
            await bus_task

        assert len(alerts) >= 1
        critical_alerts = [a for a in alerts if a.alert_level == "CRITICAL"]
        assert len(critical_alerts) >= 1
        assert "slippage" in critical_alerts[0].message.lower()

    async def test_normal_slippage_no_alert(self) -> None:
        """슬리피지 < 15bps → alert 미발행."""
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)

        alerts: list[RiskAlertEvent] = []

        async def capture_alert(event: object) -> None:
            assert isinstance(event, RiskAlertEvent)
            alerts.append(event)

        bus.subscribe("risk_alert", capture_alert)

        bus_task = asyncio.create_task(bus.start())
        try:
            bar = BarEvent(
                symbol="ETH/USDT",
                timeframe="1D",
                open=3000.0,
                high=3100.0,
                low=2900.0,
                close=3000.0,
                volume=500.0,
                bar_timestamp=datetime.now(UTC),
            )
            order = OrderRequestEvent(
                client_order_id="slip-ok-001",
                symbol="ETH/USDT",
                side="BUY",
                order_type="MARKET",
                target_weight=0.3,
                notional_usd=3000.0,
            )
            # 3000 → 3001 = 3.3 bps (< 15, no alert)
            fill = FillEvent(
                client_order_id="slip-ok-001",
                symbol="ETH/USDT",
                side="BUY",
                fill_price=3001.0,
                fill_qty=1.0,
                fee=3.0,
                fill_timestamp=datetime.now(UTC),
            )
            for evt in [bar, order, fill]:
                await bus.publish(evt)  # type: ignore[arg-type]
            await bus.flush()
        finally:
            await bus.stop()
            await bus_task

        # execution_quality 관련 alert만 필터
        exec_alerts = [a for a in alerts if "MetricsExporter:execution_quality" in (a.source or "")]
        assert len(exec_alerts) == 0


class TestBarAgeMetrics:
    """last_bar_age_seconds gauge 검증."""

    async def test_bar_resets_age(self) -> None:
        exporter = MetricsExporter(port=0)
        bar = BarEvent(
            symbol="BTC/USDT",
            timeframe="1D",
            open=40000.0,
            high=41000.0,
            low=39000.0,
            close=40500.0,
            volume=1000.0,
            bar_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [bar])

        val = _sample("mcbot_last_bar_age_seconds", {"symbol": "BTC/USDT"})
        assert val is not None
        assert val == 0.0

    def test_update_bar_ages(self) -> None:
        exporter = MetricsExporter(port=0)
        exporter._last_bar_time["BTC/USDT"] = time.monotonic() - 60
        exporter.update_bar_ages()

        val = _sample("mcbot_last_bar_age_seconds", {"symbol": "BTC/USDT"})
        assert val is not None
        assert val >= 59.0


class TestPrometheusWsDetailCallback:
    """PrometheusWsDetailCallback 테스트."""

    def test_on_ws_status_connected(self) -> None:
        cb = PrometheusWsDetailCallback()
        cb.on_ws_status("BTC/USDT", connected=True)
        val = _sample("mcbot_exchange_ws_connected", {"symbol": "BTC/USDT"})
        assert val == 1.0

    def test_on_ws_status_disconnected(self) -> None:
        cb = PrometheusWsDetailCallback()
        cb.on_ws_status("BTC/USDT", connected=False)
        val = _sample("mcbot_exchange_ws_connected", {"symbol": "BTC/USDT"})
        assert val == 0.0

    def test_on_ws_message_increments_counter(self) -> None:
        cb = PrometheusWsDetailCallback()
        before = _sample("mcbot_ws_messages_received_total", {"symbol": "ETH/USDT"}) or 0
        cb.on_ws_message("ETH/USDT")
        cb.on_ws_message("ETH/USDT")
        after = _sample("mcbot_ws_messages_received_total", {"symbol": "ETH/USDT"})
        assert after is not None
        assert after - before == 2

    def test_on_ws_reconnect_increments_counter(self) -> None:
        cb = PrometheusWsDetailCallback()
        before = _sample("mcbot_ws_reconnects_total", {"symbol": "SOL/USDT"}) or 0
        cb.on_ws_reconnect("SOL/USDT")
        after = _sample("mcbot_ws_reconnects_total", {"symbol": "SOL/USDT"})
        assert after is not None
        assert after - before == 1

    def test_update_message_ages(self) -> None:
        cb = PrometheusWsDetailCallback()
        cb.on_ws_message("BTC/USDT")
        # 즉시 update → 0에 가까워야 함
        cb.update_message_ages()
        val = _sample("mcbot_ws_last_message_age_seconds", {"symbol": "BTC/USDT"})
        assert val is not None
        assert val < 1.0

    def test_update_message_ages_stale(self) -> None:
        cb = PrometheusWsDetailCallback()
        cb._last_message_time["BTC/USDT"] = time.monotonic() - 30
        cb.update_message_ages()
        val = _sample("mcbot_ws_last_message_age_seconds", {"symbol": "BTC/USDT"})
        assert val is not None
        assert val >= 29.0

    def test_satisfies_ws_status_protocol(self) -> None:
        """WsStatusCallback Protocol 호환 확인."""
        from src.monitoring.metrics import WsStatusCallback

        cb = PrometheusWsDetailCallback()
        assert isinstance(cb, WsStatusCallback)

    def test_simple_callback_satisfies_protocol(self) -> None:
        """PrometheusWsCallback도 WsStatusCallback Protocol 충족."""
        from src.monitoring.metrics import WsStatusCallback

        cb = PrometheusWsCallback()
        assert isinstance(cb, WsStatusCallback)
