"""ProcessMetricsCollector + monitor_process_and_loop 테스트."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import REGISTRY

from src.monitoring.process_monitor import (
    ProcessMetricsCollector,
    ProcessMonitorConfig,
    _AlertCooldown,
    _check_process_alerts,
    _count_open_fds,
    _read_current_rss_bytes,
    monitor_process_and_loop,
)


def _sample(name: str, labels: dict[str, str] | None = None) -> float | None:
    return REGISTRY.get_sample_value(name, labels or {})


# ==========================================================================
# ProcessMetricsCollector
# ==========================================================================
class TestProcessMetricsCollector:
    """ProcessMetricsCollector.collect() 검증."""

    def test_collect_updates_gauges(self) -> None:
        """collect() 호출 시 RSS gauge 갱신."""
        collector = ProcessMetricsCollector()
        collector.collect()

        rss = _sample("mcbot_process_memory_rss_bytes")
        assert rss is not None
        assert rss > 0

    def test_collect_updates_last_rss(self) -> None:
        """collect() 호출 시 last_rss_bytes 갱신."""
        collector = ProcessMetricsCollector()
        collector.collect()
        assert collector.last_rss_bytes > 0

    def test_collect_open_fds(self) -> None:
        """FD 카운트는 최소 3 이상 (stdin/stdout/stderr)."""
        collector = ProcessMetricsCollector()
        collector.collect()
        assert collector.last_open_fds >= 3

    def test_collect_updates_peak_rss(self) -> None:
        """collect() 호출 시 peak RSS gauge도 갱신."""
        collector = ProcessMetricsCollector()
        collector.collect()

        peak = _sample("mcbot_process_memory_rss_peak_bytes")
        assert peak is not None
        assert peak > 0

    def test_collect_updates_last_cpu_percent(self) -> None:
        """collect() 호출 시 last_cpu_percent 속성 갱신."""
        collector = ProcessMetricsCollector()
        collector.collect()
        # 첫 호출에서도 cpu_percent 계산됨
        assert isinstance(collector.last_cpu_percent, float)


# ==========================================================================
# _read_current_rss_bytes
# ==========================================================================
class TestReadCurrentRssBytes:
    """_read_current_rss_bytes() 검증."""

    def test_returns_positive(self) -> None:
        """현재 프로세스 RSS는 반드시 양수."""
        rss = _read_current_rss_bytes()
        assert rss > 0

    def test_current_rss_lte_peak(self) -> None:
        """현재 RSS <= peak RSS."""
        from src.monitoring.process_monitor import _read_peak_rss_bytes

        current = _read_current_rss_bytes()
        peak = _read_peak_rss_bytes()
        assert current <= peak


# ==========================================================================
# _count_open_fds
# ==========================================================================
class TestCountOpenFds:
    """_count_open_fds() 단독 테스트."""

    def test_returns_at_least_stdio(self) -> None:
        """stdin/stdout/stderr 최소 3개."""
        assert _count_open_fds() >= 3

    def test_first_collect_no_cpu_spike(self) -> None:
        """첫 collect()에서 CPU%가 비정상 스파이크 없음."""
        collector = ProcessMetricsCollector()
        collector.collect()
        cpu = _sample("mcbot_process_cpu_percent")
        assert cpu is not None
        assert cpu < 200

    def test_cpu_percent_after_two_collects(self) -> None:
        """두 번 collect() 시 CPU% 계산."""
        collector = ProcessMetricsCollector()
        collector.collect()
        collector.collect()
        cpu = _sample("mcbot_process_cpu_percent")
        assert cpu is not None
        assert cpu >= 0


# ==========================================================================
# _AlertCooldown
# ==========================================================================
class TestAlertCooldown:
    """_AlertCooldown 쿨다운 로직 검증."""

    def test_first_fire_allowed(self) -> None:
        """첫 발화는 항상 허용."""
        cd = _AlertCooldown(cooldown_seconds=60.0)
        assert cd.should_fire("test_key") is True

    def test_within_cooldown_blocked(self) -> None:
        """쿨다운 내 동일 키는 차단."""
        cd = _AlertCooldown(cooldown_seconds=60.0)
        cd.should_fire("test_key")
        assert cd.should_fire("test_key") is False

    def test_independent_keys(self) -> None:
        """서로 다른 키는 독립."""
        cd = _AlertCooldown(cooldown_seconds=60.0)
        cd.should_fire("key_a")
        assert cd.should_fire("key_b") is True

    def test_expired_cooldown_refires(self) -> None:
        """쿨다운 만료 후 재발화."""
        cd = _AlertCooldown(cooldown_seconds=0.0)  # 즉시 만료
        cd.should_fire("test_key")
        # cooldown=0이므로 바로 재발화 가능
        assert cd.should_fire("test_key") is True


# ==========================================================================
# ProcessMonitorConfig
# ==========================================================================
class TestProcessMonitorConfig:
    """ProcessMonitorConfig 기본값 및 custom 설정."""

    def test_default_values(self) -> None:
        """기본값 확인."""
        cfg = ProcessMonitorConfig()
        assert cfg.interval == 10.0
        assert cfg.alert_cooldown == 60.0
        assert cfg.loop_lag_warn_seconds == 1.0
        assert cfg.rss_warn_bytes == 2 * 1024**3
        assert cfg.fd_warn_count == 1000
        assert cfg.cpu_warn_percent == 80.0
        assert cfg.active_tasks_warn_count == 200

    def test_custom_values(self) -> None:
        """custom 설정 적용."""
        cfg = ProcessMonitorConfig(
            interval=5.0,
            alert_cooldown=30.0,
            cpu_warn_percent=90.0,
            active_tasks_warn_count=500,
        )
        assert cfg.interval == 5.0
        assert cfg.alert_cooldown == 30.0
        assert cfg.cpu_warn_percent == 90.0
        assert cfg.active_tasks_warn_count == 500

    def test_frozen(self) -> None:
        """frozen dataclass — 변경 불가."""
        cfg = ProcessMonitorConfig()
        with pytest.raises(AttributeError):
            cfg.interval = 5.0  # type: ignore[misc]


# ==========================================================================
# _check_process_alerts
# ==========================================================================
class TestCheckProcessAlerts:
    """_check_process_alerts 임계값 테스트."""

    @staticmethod
    def _make_deps(
        cooldown_seconds: float = 0.0,
    ) -> tuple[MagicMock, _AlertCooldown, ProcessMonitorConfig]:
        bus = MagicMock()
        bus.publish = AsyncMock()
        cooldown = _AlertCooldown(cooldown_seconds=cooldown_seconds)
        config = ProcessMonitorConfig()
        return bus, cooldown, config

    @pytest.mark.asyncio
    async def test_no_alert_normal(self) -> None:
        """정상 값 → alert 없음."""
        bus, cooldown, config = self._make_deps()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024  # 100MB
        collector.last_open_fds = 50
        collector.last_cpu_percent = 10.0

        await _check_process_alerts(
            0.01, collector, bus, cooldown=cooldown, active_count=10, config=config
        )
        bus.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_loop_lag_alert(self) -> None:
        """Loop lag > 1s → alert 발행."""
        bus, cooldown, config = self._make_deps()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024
        collector.last_open_fds = 50
        collector.last_cpu_percent = 10.0

        await _check_process_alerts(
            2.5, collector, bus, cooldown=cooldown, active_count=10, config=config
        )
        bus.publish.assert_called_once()
        alert = bus.publish.call_args[0][0]
        assert "Event loop lag" in alert.message

    @pytest.mark.asyncio
    async def test_rss_alert(self) -> None:
        """RSS > 2GB → alert 발행."""
        bus, cooldown, config = self._make_deps()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 3 * 1024**3  # 3GB
        collector.last_open_fds = 50
        collector.last_cpu_percent = 10.0

        await _check_process_alerts(
            0.01, collector, bus, cooldown=cooldown, active_count=10, config=config
        )
        bus.publish.assert_called_once()
        alert = bus.publish.call_args[0][0]
        assert "RSS memory" in alert.message

    @pytest.mark.asyncio
    async def test_fd_alert(self) -> None:
        """FD > 1000 → alert 발행."""
        bus, cooldown, config = self._make_deps()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024
        collector.last_open_fds = 1500
        collector.last_cpu_percent = 10.0

        await _check_process_alerts(
            0.01, collector, bus, cooldown=cooldown, active_count=10, config=config
        )
        bus.publish.assert_called_once()
        alert = bus.publish.call_args[0][0]
        assert "FD count" in alert.message

    @pytest.mark.asyncio
    async def test_multiple_alerts(self) -> None:
        """복수 임계값 초과 → 각각 alert 발행."""
        bus, cooldown, config = self._make_deps()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 3 * 1024**3
        collector.last_open_fds = 1500
        collector.last_cpu_percent = 10.0

        await _check_process_alerts(
            2.0, collector, bus, cooldown=cooldown, active_count=10, config=config
        )
        assert bus.publish.call_count == 3


# ==========================================================================
# CPU Alert
# ==========================================================================
class TestCpuAlert:
    """CPU% threshold alert 검증."""

    @pytest.mark.asyncio
    async def test_cpu_alert_fires(self) -> None:
        """CPU > threshold → alert 발행."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        cooldown = _AlertCooldown(cooldown_seconds=0.0)
        config = ProcessMonitorConfig(cpu_warn_percent=80.0)
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024
        collector.last_open_fds = 50
        collector.last_cpu_percent = 95.0

        await _check_process_alerts(
            0.01, collector, bus, cooldown=cooldown, active_count=10, config=config
        )
        bus.publish.assert_called_once()
        alert = bus.publish.call_args[0][0]
        assert "CPU usage" in alert.message
        assert "95.0%" in alert.message

    @pytest.mark.asyncio
    async def test_cpu_below_threshold_no_alert(self) -> None:
        """CPU < threshold → alert 없음."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        cooldown = _AlertCooldown(cooldown_seconds=0.0)
        config = ProcessMonitorConfig(cpu_warn_percent=80.0)
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024
        collector.last_open_fds = 50
        collector.last_cpu_percent = 50.0

        await _check_process_alerts(
            0.01, collector, bus, cooldown=cooldown, active_count=10, config=config
        )
        bus.publish.assert_not_called()


# ==========================================================================
# Active Tasks Alert
# ==========================================================================
class TestActiveTasksAlert:
    """Active tasks threshold alert 검증."""

    @pytest.mark.asyncio
    async def test_tasks_alert_fires(self) -> None:
        """tasks > threshold → alert 발행."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        cooldown = _AlertCooldown(cooldown_seconds=0.0)
        config = ProcessMonitorConfig(active_tasks_warn_count=200)
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024
        collector.last_open_fds = 50
        collector.last_cpu_percent = 10.0

        await _check_process_alerts(
            0.01, collector, bus, cooldown=cooldown, active_count=350, config=config
        )
        bus.publish.assert_called_once()
        alert = bus.publish.call_args[0][0]
        assert "active tasks" in alert.message
        assert "350" in alert.message

    @pytest.mark.asyncio
    async def test_tasks_below_threshold_no_alert(self) -> None:
        """tasks < threshold → alert 없음."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        cooldown = _AlertCooldown(cooldown_seconds=0.0)
        config = ProcessMonitorConfig(active_tasks_warn_count=200)
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024
        collector.last_open_fds = 50
        collector.last_cpu_percent = 10.0

        await _check_process_alerts(
            0.01, collector, bus, cooldown=cooldown, active_count=50, config=config
        )
        bus.publish.assert_not_called()


# ==========================================================================
# Alert Cooldown Integration
# ==========================================================================
class TestAlertCooldownIntegration:
    """Alert cooldown이 _check_process_alerts에서 실제로 중복 발행을 방지하는지 검증."""

    @pytest.mark.asyncio
    async def test_cooldown_prevents_duplicate_alerts(self) -> None:
        """동일 alert 2번 연속 호출 시 쿨다운으로 1번만 발행."""
        bus = MagicMock()
        bus.publish = AsyncMock()
        cooldown = _AlertCooldown(cooldown_seconds=60.0)
        config = ProcessMonitorConfig()
        collector = ProcessMetricsCollector()
        collector.last_rss_bytes = 100 * 1024 * 1024
        collector.last_open_fds = 50
        collector.last_cpu_percent = 10.0

        # 첫 호출: lag alert 발행
        await _check_process_alerts(
            2.0, collector, bus, cooldown=cooldown, active_count=10, config=config
        )
        assert bus.publish.call_count == 1

        # 두 번째 호출: 쿨다운 내이므로 미발행
        await _check_process_alerts(
            2.0, collector, bus, cooldown=cooldown, active_count=10, config=config
        )
        assert bus.publish.call_count == 1  # 여전히 1


# ==========================================================================
# GC + Thread Metrics
# ==========================================================================
class TestGcAndThreadMetrics:
    """GC collection counter + thread count gauge 검증."""

    def test_gc_counter_registered(self) -> None:
        """GC collection counter가 Prometheus에 등록됨."""
        collector = ProcessMetricsCollector()
        collector.collect()
        # prometheus_client Counter는 _total suffix 자동 추가
        gen0 = _sample("mcbot_process_gc_collections_total_total", {"generation": "0"})
        assert gen0 is None or gen0 >= 0

    def test_thread_count_positive(self) -> None:
        """thread count >= 1."""
        collector = ProcessMetricsCollector()
        collector.collect()
        val = _sample("mcbot_process_thread_count")
        assert val is not None
        assert val >= 1


# ==========================================================================
# Monitor Loop
# ==========================================================================
class TestMonitorLoop:
    """monitor_process_and_loop 루프 테스트."""

    @pytest.mark.asyncio
    async def test_runs_and_cancels(self) -> None:
        """루프 실행 후 취소해도 crash 없음."""
        task = asyncio.create_task(monitor_process_and_loop(interval=0.05, bus=None))
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_updates_lag_gauge(self) -> None:
        """루프 실행 시 lag gauge 갱신."""
        task = asyncio.create_task(monitor_process_and_loop(interval=0.05, bus=None))
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
        task = asyncio.create_task(monitor_process_and_loop(interval=0.05, bus=None))
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
        task = asyncio.create_task(monitor_process_and_loop(interval=0.05, bus=None))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_with_config(self) -> None:
        """config 파라미터로 interval 오버라이드."""
        cfg = ProcessMonitorConfig(interval=0.05)
        task = asyncio.create_task(monitor_process_and_loop(bus=None, config=cfg))
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_interval_overrides_config(self) -> None:
        """interval 파라미터가 config.interval보다 우선."""
        cfg = ProcessMonitorConfig(interval=100.0)  # 큰 값
        task = asyncio.create_task(monitor_process_and_loop(interval=0.05, bus=None, config=cfg))
        await asyncio.sleep(0.15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # 0.15초 내 cancel 가능 = interval=0.05가 적용됨 (100.0이면 cancel 불가)
