"""MetricsExporter 테스트 — Prometheus Counter/Gauge 검증."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from prometheus_client import REGISTRY

from src.core.event_bus import EventBus
from src.core.events import (
    BalanceUpdateEvent,
    BarEvent,
    CircuitBreakerEvent,
    FillEvent,
    PositionUpdateEvent,
    RiskAlertEvent,
    SignalEvent,
)
from src.models.types import Direction
from src.monitoring.metrics import MetricsExporter


@pytest.fixture(autouse=True)
def _reset_prometheus_registry() -> None:
    """테스트 간 prometheus collector 충돌 방지.

    모듈 레벨 Gauge/Counter는 프로세스 전역이므로 값만 리셋합니다.
    """
    # Gauge는 set(0)으로, Counter는 이미 monotonic이라 검증 시 delta 사용


def _sample(name: str, labels: dict[str, str] | None = None) -> float | None:
    """REGISTRY에서 특정 metric 값을 가져옴."""
    return REGISTRY.get_sample_value(name, labels or {})


class TestMetricsExporterRegister:
    async def test_register_subscribes_events(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)
        # 핸들러가 등록되었는지 확인 (bus 내부 검증)
        assert len(bus._handlers) >= 7


class TestBalanceEvent:
    async def test_equity_and_cash_gauges(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)

        bus_task = None
        try:
            import asyncio

            bus_task = asyncio.create_task(bus.start())

            event = BalanceUpdateEvent(
                total_equity=15000.0,
                available_cash=12000.0,
                total_margin_used=3000.0,
            )
            await bus.publish(event)
            await bus.flush()

            assert _sample("mcbot_equity_usdt") == 15000.0
            assert _sample("mcbot_cash_usdt") == 12000.0
        finally:
            await bus.stop()
            if bus_task:
                await bus_task


class TestFillEvent:
    async def test_fills_counter(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)

        bus_task = None
        try:
            import asyncio

            bus_task = asyncio.create_task(bus.start())

            before = _sample("mcbot_fills_total", {"symbol": "BTC/USDT", "side": "BUY"})
            before = before or 0.0

            event = FillEvent(
                client_order_id="test-001",
                symbol="BTC/USDT",
                side="BUY",
                fill_price=40000.0,
                fill_qty=0.1,
                fee=4.0,
                fill_timestamp=datetime.now(UTC),
            )
            await bus.publish(event)
            await bus.flush()

            after = _sample("mcbot_fills_total", {"symbol": "BTC/USDT", "side": "BUY"})
            assert after is not None
            assert after > before
        finally:
            await bus.stop()
            if bus_task:
                await bus_task


class TestBarEvent:
    async def test_bars_counter(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)

        bus_task = None
        try:
            import asyncio

            bus_task = asyncio.create_task(bus.start())

            before = _sample("mcbot_bars_total", {"timeframe": "1D"})
            before = before or 0.0

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
            await bus.publish(event)
            await bus.flush()

            after = _sample("mcbot_bars_total", {"timeframe": "1D"})
            assert after is not None
            assert after > before
        finally:
            await bus.stop()
            if bus_task:
                await bus_task


class TestSignalEvent:
    async def test_signals_counter(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)

        bus_task = None
        try:
            import asyncio

            bus_task = asyncio.create_task(bus.start())

            before = _sample("mcbot_signals_total", {"symbol": "ETH/USDT"})
            before = before or 0.0

            event = SignalEvent(
                symbol="ETH/USDT",
                strategy_name="tsmom",
                direction=Direction.LONG,
                strength=0.8,
                bar_timestamp=datetime.now(UTC),
            )
            await bus.publish(event)
            await bus.flush()

            after = _sample("mcbot_signals_total", {"symbol": "ETH/USDT"})
            assert after is not None
            assert after > before
        finally:
            await bus.stop()
            if bus_task:
                await bus_task


class TestCircuitBreakerEvent:
    async def test_cb_counter(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)

        bus_task = None
        try:
            import asyncio

            bus_task = asyncio.create_task(bus.start())

            before = _sample("mcbot_circuit_breaker_total")
            before = before or 0.0

            event = CircuitBreakerEvent(
                reason="System stop-loss triggered",
                close_all_positions=True,
            )
            await bus.publish(event)
            await bus.flush()

            after = _sample("mcbot_circuit_breaker_total")
            assert after is not None
            assert after > before
        finally:
            await bus.stop()
            if bus_task:
                await bus_task


class TestPositionUpdateEvent:
    async def test_position_size_gauge(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)

        bus_task = None
        try:
            import asyncio

            bus_task = asyncio.create_task(bus.start())

            event = PositionUpdateEvent(
                symbol="SOL/USDT",
                direction=Direction.LONG,
                size=5.0,
                avg_entry_price=100.0,
                unrealized_pnl=25.0,
            )
            await bus.publish(event)
            await bus.flush()

            val = _sample("mcbot_position_size", {"symbol": "SOL/USDT"})
            assert val == 5.0
        finally:
            await bus.stop()
            if bus_task:
                await bus_task


class TestRiskAlertEvent:
    async def test_risk_alerts_counter(self) -> None:
        bus = EventBus(queue_size=100)
        exporter = MetricsExporter(port=0)
        await exporter.register(bus)

        bus_task = None
        try:
            import asyncio

            bus_task = asyncio.create_task(bus.start())

            before = _sample("mcbot_risk_alerts_total", {"level": "WARNING"})
            before = before or 0.0

            event = RiskAlertEvent(
                alert_level="WARNING",
                message="Drawdown approaching threshold",
            )
            await bus.publish(event)
            await bus.flush()

            after = _sample("mcbot_risk_alerts_total", {"level": "WARNING"})
            assert after is not None
            assert after > before
        finally:
            await bus.stop()
            if bus_task:
                await bus_task


class TestUptime:
    def test_update_uptime(self) -> None:
        import time

        exporter = MetricsExporter(port=0)
        time.sleep(0.01)
        exporter.update_uptime()
        val = _sample("mcbot_uptime_seconds")
        assert val is not None
        assert val > 0
