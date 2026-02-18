"""HealthCheckEngine 단위 테스트."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.core.event_bus import EventBus
from src.core.events import RiskAlertEvent
from src.monitoring.health_check import (
    CheckResult,
    CheckStatus,
    HealthCheckEngine,
    make_balance_consistency_check,
    make_bar_freshness_check,
    make_ws_connectivity_check,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def bus() -> EventBus:
    return EventBus(queue_size=100)


@pytest.fixture
def engine(bus: EventBus) -> HealthCheckEngine:
    return HealthCheckEngine(bus=bus)


# ── Helper check functions ────────────────────────────────────────


async def _healthy_check() -> CheckResult:
    return CheckResult(
        name="test_check",
        status=CheckStatus.HEALTHY,
        message="All good",
        timestamp=time.monotonic(),
    )


async def _degraded_check() -> CheckResult:
    return CheckResult(
        name="test_check",
        status=CheckStatus.DEGRADED,
        message="Some issue",
        timestamp=time.monotonic(),
    )


async def _critical_check() -> CheckResult:
    return CheckResult(
        name="test_check",
        status=CheckStatus.CRITICAL,
        message="System down",
        timestamp=time.monotonic(),
    )


async def _exception_check() -> CheckResult:
    msg = "Check failed"
    raise RuntimeError(msg)


# ── Tests: CheckResult ───────────────────────────────────────────


class TestCheckResult:
    """CheckResult/CheckStatus 기본 테스트."""

    def test_status_values(self) -> None:
        assert CheckStatus.HEALTHY == "healthy"
        assert CheckStatus.DEGRADED == "degraded"
        assert CheckStatus.CRITICAL == "critical"

    def test_result_frozen(self) -> None:
        result = CheckResult(name="test", status=CheckStatus.HEALTHY, message="ok", timestamp=0.0)
        with pytest.raises(AttributeError):
            result.status = CheckStatus.CRITICAL  # type: ignore[misc]


# ── Tests: Registration ──────────────────────────────────────────


class TestRegistration:
    """체크 등록 테스트."""

    def test_register_check(self, engine: HealthCheckEngine) -> None:
        engine.register_check("healthy", _healthy_check, interval=10.0)
        assert "healthy" in engine._checks
        assert engine._checks["healthy"].interval == 10.0

    def test_register_multiple(self, engine: HealthCheckEngine) -> None:
        engine.register_check("check_a", _healthy_check)
        engine.register_check("check_b", _degraded_check)
        assert len(engine._checks) == 2


# ── Tests: run_all_once ───────────────────────────────────────────


class TestRunAllOnce:
    """단일 실행 테스트."""

    async def test_all_healthy(self, engine: HealthCheckEngine) -> None:
        engine.register_check("healthy1", _healthy_check)
        engine.register_check("healthy2", _healthy_check)

        results = await engine.run_all_once()
        assert len(results) == 2
        assert all(r.status == CheckStatus.HEALTHY for r in results)

    async def test_mixed_status(self, engine: HealthCheckEngine) -> None:
        engine.register_check("healthy", _healthy_check)
        engine.register_check("degraded", _degraded_check)
        engine.register_check("critical", _critical_check)

        results = await engine.run_all_once()
        # 이름이 check 함수 내부 이름(test_check)이므로 registration name 기준으로 조회
        assert len(results) == 3

    async def test_exception_handling(self, engine: HealthCheckEngine) -> None:
        """체크 함수 예외 → CRITICAL 결과."""
        engine.register_check("failing", _exception_check)

        results = await engine.run_all_once()
        assert len(results) == 1
        assert results[0].status == CheckStatus.CRITICAL
        assert "exception" in results[0].message.lower()


# ── Tests: overall_status ─────────────────────────────────────────


class TestOverallStatus:
    """전체 상태 판정 테스트."""

    async def test_no_checks(self, engine: HealthCheckEngine) -> None:
        assert engine.overall_status() == CheckStatus.HEALTHY

    async def test_all_healthy(self, engine: HealthCheckEngine) -> None:
        engine.register_check("h1", _healthy_check)
        engine.register_check("h2", _healthy_check)
        await engine.run_all_once()
        assert engine.overall_status() == CheckStatus.HEALTHY

    async def test_degraded_overall(self, engine: HealthCheckEngine) -> None:
        engine.register_check("h1", _healthy_check)
        engine.register_check("d1", _degraded_check)
        await engine.run_all_once()
        assert engine.overall_status() == CheckStatus.DEGRADED

    async def test_critical_overrides(self, engine: HealthCheckEngine) -> None:
        engine.register_check("h1", _healthy_check)
        engine.register_check("d1", _degraded_check)
        engine.register_check("c1", _critical_check)
        await engine.run_all_once()
        assert engine.overall_status() == CheckStatus.CRITICAL


# ── Tests: Event publishing ───────────────────────────────────────


class TestEventPublishing:
    """CRITICAL → RiskAlertEvent 발행 테스트."""

    async def test_critical_publishes_alert(self, bus: EventBus, engine: HealthCheckEngine) -> None:
        engine.register_check("critical", _critical_check)
        await engine.run_all_once()

        # bus queue에 RiskAlertEvent가 있어야 함
        events: list[RiskAlertEvent] = []
        while not bus._queue.empty():
            evt = bus._queue.get_nowait()
            if isinstance(evt, RiskAlertEvent):
                events.append(evt)

        assert len(events) == 1
        assert events[0].alert_level == "CRITICAL"
        assert "HealthCheck" in events[0].message

    async def test_healthy_no_alert(self, bus: EventBus, engine: HealthCheckEngine) -> None:
        engine.register_check("healthy", _healthy_check)
        await engine.run_all_once()

        events: list[object] = []
        while not bus._queue.empty():
            evt = bus._queue.get_nowait()
            if evt is not None:
                events.append(evt)
        assert len(events) == 0

    async def test_no_bus_still_works(self) -> None:
        """bus=None일 때도 정상 동작."""
        engine = HealthCheckEngine(bus=None)
        engine.register_check("critical", _critical_check)
        results = await engine.run_all_once()
        assert results[0].status == CheckStatus.CRITICAL


# ── Tests: Start/Stop lifecycle ───────────────────────────────────


class TestLifecycle:
    """start/stop 라이프사이클 테스트."""

    async def test_start_stop(self, engine: HealthCheckEngine) -> None:
        engine.register_check("healthy", _healthy_check, interval=0.05)
        engine.start()
        assert engine._running is True

        await asyncio.sleep(0.15)  # 2~3 iterations
        await engine.stop()
        assert engine._running is False
        assert len(engine._tasks) == 0

        # 마지막 결과 존재
        status = engine.get_status()
        assert "healthy" in status
        assert status["healthy"] is not None

    async def test_double_start(self, engine: HealthCheckEngine) -> None:
        """중복 start는 무시."""
        engine.register_check("h1", _healthy_check, interval=1.0)
        engine.start()
        engine.start()  # should not create duplicate tasks
        assert len(engine._tasks) == 1
        await engine.stop()


# ── Tests: Built-in Check Factories ───────────────────────────────


class TestBarFreshnessCheck:
    """make_bar_freshness_check 테스트."""

    async def test_healthy_when_fresh(self) -> None:
        now = time.monotonic()
        check = make_bar_freshness_check(
            get_last_bar_times=lambda: {"BTC/USDT": now, "ETH/USDT": now},
            stale_threshold=120.0,
        )
        result = await check()
        assert result.status == CheckStatus.HEALTHY

    async def test_degraded_when_stale(self) -> None:
        old_time = time.monotonic() - 300  # 5 min ago
        check = make_bar_freshness_check(
            get_last_bar_times=lambda: {"BTC/USDT": old_time},
            stale_threshold=120.0,
        )
        result = await check()
        assert result.status == CheckStatus.DEGRADED
        assert "BTC/USDT" in result.message

    async def test_healthy_when_no_bars(self) -> None:
        check = make_bar_freshness_check(
            get_last_bar_times=lambda: {},
            stale_threshold=120.0,
        )
        result = await check()
        assert result.status == CheckStatus.HEALTHY


class TestWsConnectivityCheck:
    """make_ws_connectivity_check 테스트."""

    async def test_critical_when_disconnected(self) -> None:
        check = make_ws_connectivity_check(
            get_ws_connected=lambda: False,
            get_last_message_time=lambda: None,
        )
        result = await check()
        assert result.status == CheckStatus.CRITICAL

    async def test_degraded_when_timeout(self) -> None:
        old_time = time.monotonic() - 120
        check = make_ws_connectivity_check(
            get_ws_connected=lambda: True,
            get_last_message_time=lambda: old_time,
            timeout_threshold=60.0,
        )
        result = await check()
        assert result.status == CheckStatus.DEGRADED

    async def test_healthy_when_connected(self) -> None:
        check = make_ws_connectivity_check(
            get_ws_connected=lambda: True,
            get_last_message_time=lambda: time.monotonic(),
        )
        result = await check()
        assert result.status == CheckStatus.HEALTHY


class TestBalanceConsistencyCheck:
    """make_balance_consistency_check 테스트."""

    async def test_first_call_healthy(self) -> None:
        check = make_balance_consistency_check(get_equity=lambda: 10000.0)
        result = await check()
        assert result.status == CheckStatus.HEALTHY

    async def test_stable_equity(self) -> None:
        equity = [10000.0]
        check = make_balance_consistency_check(get_equity=lambda: equity[0], threshold_pct=10.0)
        await check()  # first call
        equity[0] = 10500.0  # +5%
        result = await check()
        assert result.status == CheckStatus.HEALTHY

    async def test_sudden_change(self) -> None:
        equity = [10000.0]
        check = make_balance_consistency_check(get_equity=lambda: equity[0], threshold_pct=10.0)
        await check()  # first call
        equity[0] = 8000.0  # -20%
        result = await check()
        assert result.status == CheckStatus.DEGRADED
