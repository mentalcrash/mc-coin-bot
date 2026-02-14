"""ProcessMetricsCollector + monitor_process_and_loop 테스트."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import REGISTRY

from src.monitoring.process_monitor import (
    ProcessMetricsCollector,
    _check_process_alerts,
    monitor_process_and_loop,
)


def _sample(name: str) -> float | None:
    return REGISTRY.get_sample_value(name, {})


class TestProcessMetricsCollector:
    """ProcessMetricsCollector.collect() 검증."""

    def test_collect_updates_gauges(self) -> None:
        """collect() 호출 시 RSS gauge 갱신."""
        collector = ProcessMetricsCollector()
        collector.collect()

        rss = _sample("mcbot_process_memory_rss_bytes")
        assert rss is not None
        assert rss > 0  # 현재 프로세스 RSS는 반드시 > 0

    def test_collect_updates_last_rss(self) -> None:
        """collect() 호출 시 last_rss_bytes 갱신."""
        collector = ProcessMetricsCollector()
        collector.collect()
        assert collector.last_rss_bytes > 0

    def test_collect_open_fds(self) -> None:
        """FD 카운트는 0 이상."""
        collector = ProcessMetricsCollector()
        collector.collect()
        # macOS에서는 /proc 미지원으로 0일 수 있음
        assert collector.last_open_fds >= 0

    def test_cpu_percent_after_two_collects(self) -> None:
        """두 번 collect() 시 CPU% 계산."""
        collector = ProcessMetricsCollector()
        collector.collect()
        collector.collect()
        cpu = _sample("mcbot_process_cpu_percent")
        assert cpu is not None
        assert cpu >= 0


class TestCheckProcessAlerts:
    """_check_process_alerts 임계값 테스트."""

    @pytest.mark.asyncio
    async def test_no_alert_normal(self) -> None:
        """정상 값 → alert 없음."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024  # 100MB
        collector.last_open_fds = 50

        await _check_process_alerts(0.01, collector, bus)
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_loop_lag_alert(self) -> None:
        """Loop lag > 1s → alert 발행."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024
        collector.last_open_fds = 50

        await _check_process_alerts(2.5, collector, bus)
        bus.publish.assert_called_once()
        alert = bus.publish.call_args[0][0]
        assert "Event loop lag" in alert.message

    @pytest.mark.asyncio
    async def test_rss_alert(self) -> None:
        """RSS > 2GB → alert 발행."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 3 * 1024**3  # 3GB
        collector.last_open_fds = 50

        await _check_process_alerts(0.01, collector, bus)
        bus.publish.assert_called_once()
        alert = bus.publish.call_args[0][0]
        assert "RSS memory" in alert.message

    @pytest.mark.asyncio
    async def test_fd_alert(self) -> None:
        """FD > 1000 → alert 발행."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024
        collector.last_open_fds = 1500

        await _check_process_alerts(0.01, collector, bus)
        bus.publish.assert_called_once()
        alert = bus.publish.call_args[0][0]
        assert "FD count" in alert.message

    @pytest.mark.asyncio
    async def test_multiple_alerts(self) -> None:
        """복수 임계값 초과 → 각각 alert 발행."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 3 * 1024**3
        collector.last_open_fds = 1500

        await _check_process_alerts(2.0, collector, bus)
        assert bus.publish.call_count == 3


class TestMonitorLoop:
    """monitor_process_and_loop 루프 테스트."""

    @pytest.mark.asyncio
    async def test_runs_and_cancels(self) -> None:
        """루프 실행 후 취소해도 crash 없음."""
        task = asyncio.create_task(
            monitor_process_and_loop(interval=0.05, bus=None)
        )
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_updates_lag_gauge(self) -> None:
        """루프 실행 시 lag gauge 갱신."""
        task = asyncio.create_task(
            monitor_process_and_loop(interval=0.05, bus=None)
        )
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        lag = _sample("mcbot_event_loop_lag_seconds")
        assert lag is not None
        assert lag >= 0

    @pytest.mark.asyncio
    async def test_updates_active_tasks(self) -> None:
        """루프 실행 시 active_tasks gauge 갱신."""
        task = asyncio.create_task(
            monitor_process_and_loop(interval=0.05, bus=None)
        )
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        tasks = _sample("mcbot_active_tasks")
        assert tasks is not None
        assert tasks >= 1

    @pytest.mark.asyncio
    async def test_bus_none_no_crash(self) -> None:
        """bus=None → crash 없이 동작."""
        task = asyncio.create_task(
            monitor_process_and_loop(interval=0.05, bus=None)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
