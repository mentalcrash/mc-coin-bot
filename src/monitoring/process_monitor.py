"""Process & Event Loop 모니터 — Prometheus 메트릭 + 임계값 알림.

asyncio event loop lag, active task 수, RSS 메모리, CPU%, open FDs,
GC collections, thread count를 추적하고 Prometheus gauge/counter로 export합니다.
임계값 초과 시 RiskAlertEvent를 발행합니다 (쿨다운 적용).

psutil 미사용 — stdlib(resource, os, gc, threading)만으로 수집합니다.

Rules Applied:
    - Prometheus naming: mcbot_ prefix
    - 독립 모듈: EventBus 없어도 crash 없음
"""

from __future__ import annotations

import asyncio
import gc
import os
import pathlib
import resource
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from loguru import logger
from prometheus_client import Counter, Gauge

if TYPE_CHECKING:
    from src.core.event_bus import EventBus

# ==========================================================================
# Prometheus Gauges / Counters
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
    "Process current RSS memory in bytes",
)
process_rss_peak_bytes_gauge = Gauge(
    "mcbot_process_memory_rss_peak_bytes",
    "Process peak (high-water mark) RSS memory in bytes",
)
process_cpu_percent_gauge = Gauge(
    "mcbot_process_cpu_percent",
    "Process CPU usage percent",
)
process_open_fds_gauge = Gauge(
    "mcbot_process_open_fds",
    "Number of open file descriptors",
)
gc_collections_counter = Counter(
    "mcbot_process_gc_collections_total",
    "GC collection count by generation",
    ["generation"],
)
thread_count_gauge = Gauge(
    "mcbot_process_thread_count",
    "Number of active threads",
)

# ==========================================================================
# Default 임계값 (ProcessMonitorConfig로 오버라이드 가능)
# ==========================================================================
_DEFAULT_LOOP_LAG_WARN_SECONDS = 1.0
_DEFAULT_RSS_WARN_BYTES = 2 * 1024**3  # 2GB
_DEFAULT_FD_WARN_COUNT = 1000
_DEFAULT_CPU_WARN_PERCENT = 80.0
_DEFAULT_ACTIVE_TASKS_WARN_COUNT = 100
_DEFAULT_ALERT_COOLDOWN_SECONDS = 60.0
_DEFAULT_INTERVAL = 10.0

_PAGE_SIZE: int = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


@dataclass(frozen=True)
class ProcessMonitorConfig:
    """ProcessMonitor 설정 — 임계값과 간격을 환경별로 튜닝 가능."""

    interval: float = _DEFAULT_INTERVAL
    alert_cooldown: float = _DEFAULT_ALERT_COOLDOWN_SECONDS
    loop_lag_warn_seconds: float = _DEFAULT_LOOP_LAG_WARN_SECONDS
    rss_warn_bytes: int = _DEFAULT_RSS_WARN_BYTES
    fd_warn_count: int = _DEFAULT_FD_WARN_COUNT
    cpu_warn_percent: float = _DEFAULT_CPU_WARN_PERCENT
    active_tasks_warn_count: int = _DEFAULT_ACTIVE_TASKS_WARN_COUNT


class _AlertCooldown:
    """Alert별 쿨다운 관리 — 동일 alert key의 중복 발행을 방지."""

    def __init__(self, cooldown_seconds: float = _DEFAULT_ALERT_COOLDOWN_SECONDS) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._last_fired: dict[str, float] = {}

    def should_fire(self, alert_key: str) -> bool:
        """alert_key에 대해 발행 가능 여부를 반환. 쿨다운 내이면 False."""
        now = time.monotonic()
        last = self._last_fired.get(alert_key)
        if last is not None and (now - last) < self._cooldown_seconds:
            return False
        self._last_fired[alert_key] = now
        return True


def _read_current_rss_bytes() -> int:
    """현재 RSS (Resident Set Size) 를 바이트 단위로 반환.

    Linux: /proc/self/statm field[1] * page_size (실시간 현재값)
    macOS: ru_maxrss fallback (peak, macOS에서는 /proc 없음)
    실패 시: ru_maxrss fallback + warning 로그
    """
    # Linux: /proc/self/statm 에서 현재 RSS 읽기
    if sys.platform != "darwin":
        try:
            with pathlib.Path("/proc/self/statm").open() as f:
                fields = f.readline().split()
            # field[1] = resident pages
            return int(fields[1]) * _PAGE_SIZE
        except (OSError, IndexError, ValueError):
            logger.warning("Failed to read /proc/self/statm, falling back to ru_maxrss (peak)")

    # macOS 또는 Linux fallback: ru_maxrss (peak)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss  # macOS: bytes
    return usage.ru_maxrss * 1024  # Linux: KB → bytes


def _read_peak_rss_bytes() -> int:
    """Peak (high-water mark) RSS를 바이트 단위로 반환."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss
    return usage.ru_maxrss * 1024


class ProcessMetricsCollector:
    """stdlib 기반 프로세스 메트릭 수집기 (psutil 불필요)."""

    def __init__(self) -> None:
        times = os.times()
        self._prev_times: tuple[float, float] = (times.user, times.system)
        self._prev_wall: float = time.monotonic()
        self._prev_gc_stats: list[int] = [s["collections"] for s in gc.get_stats()]
        self.last_rss_bytes: int = 0
        self.last_cpu_percent: float = 0.0
        self.last_open_fds: int = 0

    def collect(self) -> None:
        """프로세스 메트릭을 수집하여 Prometheus gauge/counter 갱신."""
        # RSS: current (실시간) + peak (high-water mark)
        self.last_rss_bytes = _read_current_rss_bytes()
        process_rss_bytes_gauge.set(self.last_rss_bytes)
        process_rss_peak_bytes_gauge.set(_read_peak_rss_bytes())

        # CPU%: user + system time delta / wall time delta
        now = time.monotonic()
        times = os.times()
        user, system = times.user, times.system
        wall_delta = now - self._prev_wall
        if wall_delta > 0:
            cpu_delta = (user - self._prev_times[0]) + (system - self._prev_times[1])
            self.last_cpu_percent = (cpu_delta / wall_delta) * 100
            process_cpu_percent_gauge.set(self.last_cpu_percent)
        self._prev_times = (user, system)
        self._prev_wall = now

        # Open FDs: Linux /proc/self/fd, macOS /dev/fd
        self.last_open_fds = _count_open_fds()
        process_open_fds_gauge.set(self.last_open_fds)

        # GC collections: generation별 delta
        current_gc = [s["collections"] for s in gc.get_stats()]
        for gen, (cur, prev) in enumerate(zip(current_gc, self._prev_gc_stats, strict=True)):
            delta = cur - prev
            if delta > 0:
                gc_collections_counter.labels(generation=str(gen)).inc(delta)
        self._prev_gc_stats = current_gc

        # Thread count
        thread_count_gauge.set(threading.active_count())


def _count_open_fds() -> int:
    """열려 있는 파일 디스크립터 수를 반환."""
    from pathlib import Path

    fd_dir = Path("/dev/fd") if sys.platform == "darwin" else Path("/proc/self/fd")
    try:
        return len(list(fd_dir.iterdir()))
    except OSError:
        logger.warning("Failed to count open FDs from {}", fd_dir)
        return 0


async def _check_process_alerts(
    lag: float,
    collector: ProcessMetricsCollector,
    bus: EventBus,
    *,
    cooldown: _AlertCooldown,
    active_count: int,
    config: ProcessMonitorConfig,
) -> None:
    """임계값 초과 시 RiskAlertEvent 발행 (쿨다운 적용)."""
    from src.core.events import RiskAlertEvent

    alerts: list[tuple[str, Literal["WARNING", "CRITICAL"], str]] = []

    if lag > config.loop_lag_warn_seconds:
        alerts.append(
            (
                "loop_lag",
                "WARNING",
                f"Event loop lag {lag:.2f}s (threshold {config.loop_lag_warn_seconds}s)",
            )
        )

    if collector.last_rss_bytes > config.rss_warn_bytes:
        rss_gb = collector.last_rss_bytes / (1024**3)
        threshold_gb = config.rss_warn_bytes / (1024**3)
        alerts.append(
            (
                "rss_high",
                "WARNING",
                f"High RSS memory {rss_gb:.1f}GB (threshold {threshold_gb:.0f}GB)",
            )
        )

    if collector.last_open_fds > config.fd_warn_count:
        alerts.append(
            (
                "fd_high",
                "WARNING",
                f"High FD count {collector.last_open_fds} (threshold {config.fd_warn_count})",
            )
        )

    if collector.last_cpu_percent > config.cpu_warn_percent:
        alerts.append(
            (
                "cpu_high",
                "WARNING",
                f"High CPU usage {collector.last_cpu_percent:.1f}% (threshold {config.cpu_warn_percent}%)",
            )
        )

    if active_count > config.active_tasks_warn_count:
        alerts.append(
            (
                "tasks_high",
                "WARNING",
                f"High active tasks {active_count} (threshold {config.active_tasks_warn_count})",
            )
        )

    for key, level, message in alerts:
        if cooldown.should_fire(key):
            alert = RiskAlertEvent(
                alert_level=level,
                message=message,
                source="ProcessMonitor",
            )
            await bus.publish(alert)


async def monitor_process_and_loop(
    interval: float | None = None,
    bus: EventBus | None = None,
    *,
    config: ProcessMonitorConfig | None = None,
) -> None:
    """Event loop lag 측정 + 프로세스 메트릭 갱신 루프.

    lag = (monotonic after sleep) - (monotonic before) - interval

    Args:
        interval: 측정 주기 (초). None이면 config.interval 사용.
        bus: EventBus (None이면 alert 미발행)
        config: ProcessMonitorConfig (None이면 기본값)
    """
    if config is None:
        config = ProcessMonitorConfig()
    effective_interval = interval if interval is not None else config.interval

    collector = ProcessMetricsCollector()
    cooldown = _AlertCooldown(config.alert_cooldown)
    logger.info("ProcessMonitor started (interval={}s)", effective_interval)

    while True:
        t0 = time.monotonic()
        await asyncio.sleep(effective_interval)
        lag = time.monotonic() - t0 - effective_interval
        lag = max(lag, 0.0)

        event_loop_lag_gauge.set(lag)
        active_count = len(asyncio.all_tasks())
        active_tasks_gauge.set(active_count)
        collector.collect()

        if bus is not None:
            await _check_process_alerts(
                lag,
                collector,
                bus,
                cooldown=cooldown,
                active_count=active_count,
                config=config,
            )
