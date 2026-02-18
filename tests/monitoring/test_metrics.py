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
    LiveExecutorMetrics,
    MetricsExporter,
    PrometheusApiCallback,
    PrometheusLiveExecutorMetrics,
    PrometheusWsCallback,
    PrometheusWsDetailCallback,
    _calculate_signed_slippage_bps,
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
        """last_price 기반 MTM notional 확인."""
        exporter = MetricsExporter(port=0)
        event = PositionUpdateEvent(
            symbol="BTC/USDT",
            direction=Direction.LONG,
            size=0.5,
            avg_entry_price=60000.0,
            unrealized_pnl=500.0,
            last_price=62000.0,
        )
        await _run_with_bus(exporter, [event])

        notional = _sample("mcbot_position_notional_usdt", {"symbol": "BTC/USDT"})
        assert notional == 31000.0  # 0.5 * 62000 (MTM)

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
        """realized_pnl_delta 기반 증분 카운터 확인."""
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_realized_profit_usdt_total", {"symbol": "BTC/USDT"}) or 0.0

        event = PositionUpdateEvent(
            symbol="BTC/USDT",
            direction=Direction.NEUTRAL,
            size=0.0,
            avg_entry_price=0.0,
            realized_pnl=250.0,
            realized_pnl_delta=250.0,
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_realized_profit_usdt_total", {"symbol": "BTC/USDT"})
        assert after is not None
        assert after >= before + 250.0

    async def test_realized_loss_counter(self) -> None:
        """realized_pnl_delta 기반 증분 카운터 확인 (손실)."""
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_realized_loss_usdt_total", {"symbol": "BTC/USDT"}) or 0.0

        event = PositionUpdateEvent(
            symbol="BTC/USDT",
            direction=Direction.NEUTRAL,
            size=0.0,
            avg_entry_price=0.0,
            realized_pnl=-300.0,
            realized_pnl_delta=-300.0,
        )
        await _run_with_bus(exporter, [event])

        after = _sample("mcbot_realized_loss_usdt_total", {"symbol": "BTC/USDT"})
        assert after is not None
        assert after >= before + 300.0

    async def test_position_notional_fallback_to_entry(self) -> None:
        """last_price=0 시 entry price fallback."""
        exporter = MetricsExporter(port=0)
        event = PositionUpdateEvent(
            symbol="SOL/USDT",
            direction=Direction.LONG,
            size=10.0,
            avg_entry_price=150.0,
            last_price=0.0,  # fallback 조건
        )
        await _run_with_bus(exporter, [event])

        notional = _sample("mcbot_position_notional_usdt", {"symbol": "SOL/USDT"})
        assert notional == 1500.0  # 10 * 150 (entry price fallback)

    async def test_realized_pnl_no_double_counting(self) -> None:
        """B1 회귀 테스트: 2회 Fill → counter=250 (not 350)."""
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_realized_profit_usdt_total", {"symbol": "AVAX/USDT"}) or 0.0

        # Fill 1: delta=100
        event1 = PositionUpdateEvent(
            symbol="AVAX/USDT",
            direction=Direction.LONG,
            size=5.0,
            avg_entry_price=30.0,
            realized_pnl=100.0,
            realized_pnl_delta=100.0,
        )
        # Fill 2: delta=150 (누적 realized=250이지만 delta만 반영)
        event2 = PositionUpdateEvent(
            symbol="AVAX/USDT",
            direction=Direction.LONG,
            size=3.0,
            avg_entry_price=30.0,
            realized_pnl=250.0,
            realized_pnl_delta=150.0,
        )
        await _run_with_bus(exporter, [event1, event2])

        after = _sample("mcbot_realized_profit_usdt_total", {"symbol": "AVAX/USDT"})
        assert after is not None
        # counter = before + 100 + 150 = before + 250 (NOT before + 100 + 250 = before + 350)
        assert after == pytest.approx(before + 250.0, abs=0.01)


class TestBalanceUpdateExtended:
    """BalanceUpdateEvent 신규 필드 (drawdown, positions, leverage) 검증."""

    async def test_drawdown_gauge(self) -> None:
        """drawdown_pct=0.10 → gauge=10.0 (0-100% scale)."""
        exporter = MetricsExporter(port=0)
        event = BalanceUpdateEvent(
            total_equity=9000.0,
            available_cash=5000.0,
            total_margin_used=4000.0,
            drawdown_pct=0.10,
        )
        await _run_with_bus(exporter, [event])

        val = _sample("mcbot_drawdown_pct")
        assert val == pytest.approx(10.0)

    async def test_open_positions_gauge(self) -> None:
        """open_position_count=3 → gauge=3.0."""
        exporter = MetricsExporter(port=0)
        event = BalanceUpdateEvent(
            total_equity=10000.0,
            available_cash=7000.0,
            open_position_count=3,
        )
        await _run_with_bus(exporter, [event])

        val = _sample("mcbot_open_positions")
        assert val == 3.0

    async def test_aggregate_leverage_gauge(self) -> None:
        """aggregate_leverage=1.5 → gauge=1.5."""
        exporter = MetricsExporter(port=0)
        event = BalanceUpdateEvent(
            total_equity=10000.0,
            available_cash=5000.0,
            aggregate_leverage=1.5,
        )
        await _run_with_bus(exporter, [event])

        val = _sample("mcbot_aggregate_leverage")
        assert val == 1.5


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


class TestSignedSlippageCalculation:
    """_calculate_signed_slippage_bps() 단위 테스트."""

    def test_buy_adverse(self) -> None:
        """BUY: fill > expected → 양수 (불리)."""
        bps = _calculate_signed_slippage_bps(40000.0, 40020.0, "BUY")
        assert bps == pytest.approx(5.0)

    def test_buy_favorable(self) -> None:
        """BUY: fill < expected → 음수 (유리)."""
        bps = _calculate_signed_slippage_bps(40000.0, 39980.0, "BUY")
        assert bps == pytest.approx(-5.0)

    def test_sell_adverse(self) -> None:
        """SELL: fill < expected → 양수 (불리)."""
        bps = _calculate_signed_slippage_bps(40000.0, 39980.0, "SELL")
        assert bps == pytest.approx(5.0)

    def test_sell_favorable(self) -> None:
        """SELL: fill > expected → 음수 (유리)."""
        bps = _calculate_signed_slippage_bps(40000.0, 40020.0, "SELL")
        assert bps == pytest.approx(-5.0)

    def test_zero_slippage(self) -> None:
        """체결가 == 기대가 → 0."""
        assert _calculate_signed_slippage_bps(100.0, 100.0, "BUY") == 0.0
        assert _calculate_signed_slippage_bps(100.0, 100.0, "SELL") == 0.0

    def test_zero_expected_price(self) -> None:
        """기대가 0 → 0."""
        assert _calculate_signed_slippage_bps(0.0, 100.0, "BUY") == 0.0


class TestSignedSlippageMetric:
    """mcbot_slippage_signed_bps Histogram 검증."""

    async def test_signed_slippage_observed(self) -> None:
        """Fill 시 signed slippage가 observe됨."""
        exporter = MetricsExporter(port=0)
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
            client_order_id="signed-001",
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        fill = FillEvent(
            client_order_id="signed-001",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=40020.0,
            fill_qty=0.1,
            fee=4.0,
            fill_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [bar, order, fill])

        # signed histogram bucket에 값이 기록되었는지 확인
        count = _sample(
            "mcbot_slippage_signed_bps_count",
            {"symbol": "BTC/USDT", "side": "BUY"},
        )
        assert count is not None
        assert count > 0


class TestRejectionReasonExtended:
    """보강된 _categorize_reason() 테스트."""

    def test_aggregate_leverage(self) -> None:
        assert _categorize_reason("Aggregate leverage exceeded: 5.0x") == "leverage_exceeded"

    def test_positions_reached(self) -> None:
        assert _categorize_reason("Max positions reached (8)") == "max_positions"

    def test_order_size_space(self) -> None:
        assert _categorize_reason("Order size too large for BTC/USDT") == "order_size_exceeded"

    def test_duplicate(self) -> None:
        assert _categorize_reason("Duplicate order") == "duplicate"


class TestLiveExecutorMetricsProtocol:
    """LiveExecutorMetrics Protocol + Prometheus 구현 검증."""

    def test_protocol_satisfies(self) -> None:
        """PrometheusLiveExecutorMetrics가 Protocol을 충족."""
        cb = PrometheusLiveExecutorMetrics()
        assert isinstance(cb, LiveExecutorMetrics)

    def test_min_notional_skip_counter(self) -> None:
        cb = PrometheusLiveExecutorMetrics()
        before = _sample("mcbot_live_min_notional_skip_total", {"symbol": "BTC/USDT"}) or 0
        cb.on_min_notional_skip("BTC/USDT")
        after = _sample("mcbot_live_min_notional_skip_total", {"symbol": "BTC/USDT"})
        assert after is not None
        assert after - before == 1

    def test_api_blocked_counter(self) -> None:
        cb = PrometheusLiveExecutorMetrics()
        before = _sample("mcbot_live_api_blocked_total", {"symbol": "ETH/USDT"}) or 0
        cb.on_api_blocked("ETH/USDT")
        after = _sample("mcbot_live_api_blocked_total", {"symbol": "ETH/USDT"})
        assert after is not None
        assert after - before == 1

    def test_partial_fill_counter(self) -> None:
        cb = PrometheusLiveExecutorMetrics()
        before = _sample("mcbot_live_partial_fill_total", {"symbol": "SOL/USDT"}) or 0
        cb.on_partial_fill("SOL/USDT")
        after = _sample("mcbot_live_partial_fill_total", {"symbol": "SOL/USDT"})
        assert after is not None
        assert after - before == 1

    def test_fill_parse_failure_counter(self) -> None:
        cb = PrometheusLiveExecutorMetrics()
        before = _sample("mcbot_live_fill_parse_failure_total", {"symbol": "DOGE/USDT"}) or 0
        cb.on_fill_parse_failure("DOGE/USDT")
        after = _sample("mcbot_live_fill_parse_failure_total", {"symbol": "DOGE/USDT"})
        assert after is not None
        assert after - before == 1


class TestStrategySlippage:
    """Per-strategy slippage 메트릭 검증."""

    async def test_strategy_slippage_observed(self) -> None:
        """Fill 시 per-strategy slippage가 observe됨."""
        exporter = MetricsExporter(port=0)
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
            client_order_id="ctrend-BTCUSDT-50",
            symbol="BTC/USDT",
            side="BUY",
            order_type="MARKET",
            target_weight=0.5,
            notional_usd=5000.0,
        )
        fill = FillEvent(
            client_order_id="ctrend-BTCUSDT-50",
            symbol="BTC/USDT",
            side="BUY",
            fill_price=40010.0,
            fill_qty=0.1,
            fee=4.0,
            fill_timestamp=datetime.now(UTC),
        )
        await _run_with_bus(exporter, [bar, order, fill])

        count = _sample(
            "mcbot_strategy_slippage_bps_count",
            {"strategy": "ctrend", "symbol": "BTC/USDT", "side": "BUY"},
        )
        assert count is not None
        assert count > 0


class TestStrategyPnlAttribution:
    """Per-strategy PnL attribution 검증 (Step 1-3)."""

    async def test_strategy_pnl_attribution(self) -> None:
        """Fill → PositionUpdate 체인에서 PnL이 올바른 전략에 귀속."""
        exporter = MetricsExporter(port=0)

        order = OrderRequestEvent(
            client_order_id="ctrend-BTCUSDT-10",
            symbol="BTC/USDT",
            side="SELL",
            order_type="MARKET",
            target_weight=0.0,
            notional_usd=5000.0,
        )
        fill = FillEvent(
            client_order_id="ctrend-BTCUSDT-10",
            symbol="BTC/USDT",
            side="SELL",
            fill_price=42000.0,
            fill_qty=0.1,
            fee=4.2,
            fill_timestamp=datetime.now(UTC),
        )
        position = PositionUpdateEvent(
            symbol="BTC/USDT",
            direction=Direction.NEUTRAL,
            size=0.0,
            avg_entry_price=0.0,
            realized_pnl=200.0,
            realized_pnl_delta=200.0,
        )
        await _run_with_bus(exporter, [order, fill, position])

        pnl = _sample("mcbot_strategy_pnl_usdt", {"strategy": "ctrend"})
        assert pnl == pytest.approx(200.0)

    async def test_strategy_pnl_cumulative(self) -> None:
        """여러 fill에 걸친 누적 PnL gauge 정확성."""
        exporter = MetricsExporter(port=0)

        events: list[object] = []
        # Trade 1: +100
        events.append(
            OrderRequestEvent(
                client_order_id="ctrend-BTCUSDT-20",
                symbol="BTC/USDT",
                side="SELL",
                order_type="MARKET",
                target_weight=0.0,
                notional_usd=3000.0,
            )
        )
        events.append(
            FillEvent(
                client_order_id="ctrend-BTCUSDT-20",
                symbol="BTC/USDT",
                side="SELL",
                fill_price=41000.0,
                fill_qty=0.1,
                fee=4.0,
                fill_timestamp=datetime.now(UTC),
            )
        )
        events.append(
            PositionUpdateEvent(
                symbol="BTC/USDT",
                direction=Direction.NEUTRAL,
                size=0.0,
                avg_entry_price=0.0,
                realized_pnl=100.0,
                realized_pnl_delta=100.0,
            )
        )
        # Trade 2: -50
        events.append(
            OrderRequestEvent(
                client_order_id="ctrend-BTCUSDT-21",
                symbol="BTC/USDT",
                side="SELL",
                order_type="MARKET",
                target_weight=0.0,
                notional_usd=3000.0,
            )
        )
        events.append(
            FillEvent(
                client_order_id="ctrend-BTCUSDT-21",
                symbol="BTC/USDT",
                side="SELL",
                fill_price=39000.0,
                fill_qty=0.1,
                fee=4.0,
                fill_timestamp=datetime.now(UTC),
            )
        )
        events.append(
            PositionUpdateEvent(
                symbol="BTC/USDT",
                direction=Direction.NEUTRAL,
                size=0.0,
                avg_entry_price=0.0,
                realized_pnl=50.0,
                realized_pnl_delta=-50.0,
            )
        )
        await _run_with_bus(exporter, events)

        # Cumulative = +100 + (-50) = +50
        pnl = _sample("mcbot_strategy_pnl_usdt", {"strategy": "ctrend"})
        assert pnl == pytest.approx(50.0)

    async def test_strategy_realized_profit_loss_split(self) -> None:
        """profit/loss counter 분리 정확성."""
        exporter = MetricsExporter(port=0)
        before_profit = (
            _sample("mcbot_strategy_realized_profit_usdt_total", {"strategy": "anchor-mom"}) or 0.0
        )
        before_loss = (
            _sample("mcbot_strategy_realized_loss_usdt_total", {"strategy": "anchor-mom"}) or 0.0
        )

        events: list[object] = []
        # Profit trade: +300
        events.append(
            OrderRequestEvent(
                client_order_id="anchor-mom-ETHUSDT-1",
                symbol="ETH/USDT",
                side="SELL",
                order_type="MARKET",
                target_weight=0.0,
                notional_usd=3000.0,
            )
        )
        events.append(
            FillEvent(
                client_order_id="anchor-mom-ETHUSDT-1",
                symbol="ETH/USDT",
                side="SELL",
                fill_price=3300.0,
                fill_qty=1.0,
                fee=3.3,
                fill_timestamp=datetime.now(UTC),
            )
        )
        events.append(
            PositionUpdateEvent(
                symbol="ETH/USDT",
                direction=Direction.NEUTRAL,
                size=0.0,
                avg_entry_price=0.0,
                realized_pnl=300.0,
                realized_pnl_delta=300.0,
            )
        )
        # Loss trade: -120
        events.append(
            OrderRequestEvent(
                client_order_id="anchor-mom-ETHUSDT-2",
                symbol="ETH/USDT",
                side="SELL",
                order_type="MARKET",
                target_weight=0.0,
                notional_usd=3000.0,
            )
        )
        events.append(
            FillEvent(
                client_order_id="anchor-mom-ETHUSDT-2",
                symbol="ETH/USDT",
                side="SELL",
                fill_price=2900.0,
                fill_qty=1.0,
                fee=2.9,
                fill_timestamp=datetime.now(UTC),
            )
        )
        events.append(
            PositionUpdateEvent(
                symbol="ETH/USDT",
                direction=Direction.NEUTRAL,
                size=0.0,
                avg_entry_price=0.0,
                realized_pnl=180.0,
                realized_pnl_delta=-120.0,
            )
        )
        await _run_with_bus(exporter, events)

        after_profit = _sample(
            "mcbot_strategy_realized_profit_usdt_total", {"strategy": "anchor-mom"}
        )
        after_loss = _sample(
            "mcbot_strategy_realized_loss_usdt_total", {"strategy": "anchor-mom"}
        )
        assert after_profit is not None
        assert after_profit >= before_profit + 300.0
        assert after_loss is not None
        assert after_loss >= before_loss + 120.0

    async def test_strategy_trade_count(self) -> None:
        """trade_count가 realized PnL 발생 시에만 증가."""
        exporter = MetricsExporter(port=0)
        before = _sample("mcbot_strategy_trade_count_total", {"strategy": "ctrend"}) or 0.0

        events: list[object] = []
        # Fill without realized PnL (position open)
        events.append(
            OrderRequestEvent(
                client_order_id="ctrend-SOLUSDT-1",
                symbol="SOL/USDT",
                side="BUY",
                order_type="MARKET",
                target_weight=0.5,
                notional_usd=2000.0,
            )
        )
        events.append(
            FillEvent(
                client_order_id="ctrend-SOLUSDT-1",
                symbol="SOL/USDT",
                side="BUY",
                fill_price=100.0,
                fill_qty=10.0,
                fee=1.0,
                fill_timestamp=datetime.now(UTC),
            )
        )
        events.append(
            PositionUpdateEvent(
                symbol="SOL/USDT",
                direction=Direction.LONG,
                size=10.0,
                avg_entry_price=100.0,
                realized_pnl_delta=0.0,
            )
        )
        # Fill with realized PnL (position close)
        events.append(
            OrderRequestEvent(
                client_order_id="ctrend-SOLUSDT-2",
                symbol="SOL/USDT",
                side="SELL",
                order_type="MARKET",
                target_weight=0.0,
                notional_usd=2000.0,
            )
        )
        events.append(
            FillEvent(
                client_order_id="ctrend-SOLUSDT-2",
                symbol="SOL/USDT",
                side="SELL",
                fill_price=110.0,
                fill_qty=10.0,
                fee=1.1,
                fill_timestamp=datetime.now(UTC),
            )
        )
        events.append(
            PositionUpdateEvent(
                symbol="SOL/USDT",
                direction=Direction.NEUTRAL,
                size=0.0,
                avg_entry_price=0.0,
                realized_pnl=100.0,
                realized_pnl_delta=100.0,
            )
        )
        await _run_with_bus(exporter, events)

        after = _sample("mcbot_strategy_trade_count_total", {"strategy": "ctrend"})
        assert after is not None
        # Only the close fill should increment trade_count (delta=0 skipped)
        assert after == pytest.approx(before + 1.0)

    async def test_strategy_pnl_no_pending(self) -> None:
        """_pending_orders 미매칭 fill 시 PnL gauge 변화 없음."""
        exporter = MetricsExporter(port=0)

        # Fill without prior OrderRequest (no pending match)
        fill = FillEvent(
            client_order_id="unknown-order-123",
            symbol="DOGE/USDT",
            side="BUY",
            fill_price=0.1,
            fill_qty=10000.0,
            fee=1.0,
            fill_timestamp=datetime.now(UTC),
        )
        position = PositionUpdateEvent(
            symbol="DOGE/USDT",
            direction=Direction.NEUTRAL,
            size=0.0,
            avg_entry_price=0.0,
            realized_pnl=50.0,
            realized_pnl_delta=50.0,
        )
        await _run_with_bus(exporter, [fill, position])

        # _last_fill_strategy never set for DOGE/USDT → no strategy PnL gauge update
        assert exporter._strategy_cumulative_pnl == {}


class TestErrorsCounter:
    """mcbot_errors_total counter 검증 (A-1: Dead Code 수정)."""

    async def test_eventbus_handler_error_increments_counter(self) -> None:
        """EventBus handler error 시 errors_counter가 증가."""
        before = (
            _sample(
                "mcbot_errors_total",
                {"component": "EventBus", "error_type": "ValueError"},
            )
            or 0.0
        )

        bus = EventBus(queue_size=100)

        async def failing_handler(_event: object) -> None:
            raise ValueError("test error")

        from src.core.events import EventType

        bus.subscribe(EventType.BAR, failing_handler)
        bus_task = asyncio.create_task(bus.start())
        try:
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
            await bus.publish(bar)
            await bus.flush()
        finally:
            await bus.stop()
            await bus_task

        after = _sample(
            "mcbot_errors_total",
            {"component": "EventBus", "error_type": "ValueError"},
        )
        assert after is not None
        assert after > before

    def test_errors_counter_direct_inc(self) -> None:
        """errors_counter를 직접 호출해서 라벨 동작 확인."""
        from src.monitoring.metrics import errors_counter

        before = (
            _sample(
                "mcbot_errors_total",
                {"component": "LiveRunner", "error_type": "RuntimeError"},
            )
            or 0.0
        )
        errors_counter.labels(component="LiveRunner", error_type="RuntimeError").inc()
        after = _sample(
            "mcbot_errors_total",
            {"component": "LiveRunner", "error_type": "RuntimeError"},
        )
        assert after is not None
        assert after > before


class TestHeartbeatEventPublish:
    """HeartbeatEvent 발행 검증 (A-2: LiveRunner에서 발행)."""

    async def test_heartbeat_event_updates_gauge(self) -> None:
        """HeartbeatEvent 발행 → heartbeat_timestamp gauge 갱신."""
        exporter = MetricsExporter(port=0)
        now = datetime.now(UTC)
        event = HeartbeatEvent(component="LiveRunner", timestamp=now)
        await _run_with_bus(exporter, [event])

        val = _sample("mcbot_heartbeat_timestamp")
        assert val is not None
        assert val == pytest.approx(now.timestamp(), abs=1.0)

    async def test_heartbeat_component_field(self) -> None:
        """HeartbeatEvent의 component 필드가 올바르게 설정됨."""
        event = HeartbeatEvent(component="LiveRunner")
        assert event.component == "LiveRunner"
        assert event.event_type.value == "heartbeat"
