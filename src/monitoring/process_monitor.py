"""Process & Event Loop 모니터 — Prometheus 메트릭 + 임계값 알림.

asyncio event loop lag, active task 수, RSS 메모리, CPU%, open FDs를 추적하고
Prometheus gauge로 export합니다. 임계값 초과 시 RiskAlertEvent를 발행합니다.

psutil 미사용 — stdlib(resource, os)만으로 수집합니다.

Rules Applied:
    - Prometheus naming: mcbot_ prefix
    - 독립 모듈: EventBus 없어도 crash 없음
"""

from __future__ import annotations

import asyncio
import os
import resource
import sys
import time
from typing import TYPE_CHECKING, Literal

from loguru import logger
from prometheus_client import Gauge

if TYPE_CHECKING:
    from src.core.event_bus import EventBus

# ==========================================================================
# Prometheus Gauges
# ==========================================================================
event_loop_lag_gauge = Gauge(
    "mcbot_event_loop_lag_seconds",
    "Event loop scheduling lag",
)
active_tasks_gauge = Gauge(
    "mcbot_active_tasks",
    "Number of active asyncio tasks",
)
process_rss_bytes_gauge = Gauge(
    "mcbot_process_memory_rss_bytes",
    "Process RSS memory in bytes",
)
process_cpu_percent_gauge = Gauge(
    "mcbot_process_cpu_percent",
    "Process CPU usage percent",
)
process_open_fds_gauge = Gauge(
    "mcbot_process_open_fds",
    "Number of open file descriptors",
)

# ==========================================================================
# 임계값
# ==========================================================================
_LOOP_LAG_WARN_SECONDS = 1.0
_RSS_WARN_BYTES = 2 * 1024**3  # 2GB
_FD_WARN_COUNT = 1000


class ProcessMetricsCollector:
    """stdlib 기반 프로세스 메트릭 수집기 (psutil 불필요)."""

    def __init__(self) -> None:
        times = os.times()
        self._prev_times: tuple[float, float] = (times.user, times.system)
        self._prev_wall: float = time.monotonic()
        self.last_rss_bytes: int = 0
        self.last_open_fds: int = 0

    def collect(self) -> None:
        """프로세스 메트릭을 수집하여 Prometheus gauge 갱신."""
        # RSS: ru_maxrss는 macOS에서 bytes, Linux에서 KB
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            self.last_rss_bytes = usage.ru_maxrss
        else:
            self.last_rss_bytes = usage.ru_maxrss * 1024
        process_rss_bytes_gauge.set(self.last_rss_bytes)

        # CPU%: user + system time delta / wall time delta
        now = time.monotonic()
        times = os.times()
        user, system = times.user, times.system
        wall_delta = now - self._prev_wall
        if wall_delta > 0:
            cpu_delta = (user - self._prev_times[0]) + (system - self._prev_times[1])
            cpu_pct = (cpu_delta / wall_delta) * 100
            process_cpu_percent_gauge.set(cpu_pct)
        self._prev_times = (user, system)
        self._prev_wall = now

        # Open FDs: Linux /proc/self/fd, macOS fallback
        self.last_open_fds = _count_open_fds()
        process_open_fds_gauge.set(self.last_open_fds)


def _count_open_fds() -> int:
    """열려 있는 파일 디스크립터 수를 반환."""
    from pathlib import Path

    fd_dir = Path("/dev/fd") if sys.platform == "darwin" else Path("/proc/self/fd")
    try:
        return len(list(fd_dir.iterdir()))
    except OSError:
        return 0


async def _check_process_alerts(
    lag: float,
    collector: ProcessMetricsCollector,
    bus: EventBus,
) -> None:
    """임계값 초과 시 RiskAlertEvent 발행."""
    from src.core.events import RiskAlertEvent

    alerts: list[tuple[Literal["WARNING", "CRITICAL"], str]] = []

    if lag > _LOOP_LAG_WARN_SECONDS:
        alerts.append(
            ("WARNING", f"Event loop lag {lag:.2f}s (threshold {_LOOP_LAG_WARN_SECONDS}s)")
        )

    if collector.last_rss_bytes > _RSS_WARN_BYTES:
        rss_gb = collector.last_rss_bytes / (1024**3)
        alerts.append(("WARNING", f"High RSS memory {rss_gb:.1f}GB (threshold 2GB)"))

    if collector.last_open_fds > _FD_WARN_COUNT:
        alerts.append(
            ("WARNING", f"High FD count {collector.last_open_fds} (threshold {_FD_WARN_COUNT})")
        )

    for level, message in alerts:
        alert = RiskAlertEvent(
            alert_level=level,
            message=message,
            source="ProcessMonitor",
        )
        await bus.publish(alert)


async def monitor_process_and_loop(
    interval: float = 10.0,
    bus: EventBus | None = None,
) -> None:
    """Event loop lag 측정 + 프로세스 메트릭 갱신 루프.

    lag = (monotonic after sleep) - (monotonic before) - interval

    Args:
        interval: 측정 주기 (초)
        bus: EventBus (None이면 alert 미발행)
    """
    collector = ProcessMetricsCollector()
    logger.info("ProcessMonitor started (interval={}s)", interval)

    while True:
        t0 = time.monotonic()
        await asyncio.sleep(interval)
        lag = time.monotonic() - t0 - interval
        lag = max(lag, 0.0)

        event_loop_lag_gauge.set(lag)
        active_tasks_gauge.set(len(asyncio.all_tasks()))
        collector.collect()

        if bus is not None:
            await _check_process_alerts(lag, collector, bus)
