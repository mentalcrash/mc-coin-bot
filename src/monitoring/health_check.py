"""Health Check Engine -- 자동화된 시스템 상태 진단.

Plugin 패턴으로 확장 가능한 Health Check 엔진.
주기적으로 등록된 check 함수를 실행하여 시스템 상태를 판정합니다.

CRITICAL 상태 시 RiskAlertEvent를 발행합니다.

Rules Applied:
    - Plugin 패턴: register_check(name, check_fn, interval)
    - RiskAlertEvent 재사용 (새 이벤트 타입 불필요)
    - asyncio.create_task로 주기적 실행
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import RiskAlertEvent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.core.event_bus import EventBus


class CheckStatus(StrEnum):
    """헬스 체크 결과 상태."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass(frozen=True)
class CheckResult:
    """개별 체크 실행 결과.

    Attributes:
        name: 체크 이름
        status: 결과 상태
        message: 상세 메시지
        timestamp: 실행 시각 (monotonic)
    """

    name: str
    status: CheckStatus
    message: str
    timestamp: float


@dataclass
class HealthCheckRegistration:
    """등록된 체크 메타데이터.

    Attributes:
        name: 체크 이름
        check_fn: async 체크 함수 (CheckResult 반환)
        interval: 실행 주기 (초)
        last_result: 마지막 실행 결과
    """

    name: str
    check_fn: Callable[[], Awaitable[CheckResult]]
    interval: float
    last_result: CheckResult | None = field(default=None)


class HealthCheckEngine:
    """자동화된 Health Check 엔진.

    ``register_check()``로 체크 함수를 등록하고,
    ``start()``로 주기적 실행을 시작합니다.
    CRITICAL 체크 결과 시 RiskAlertEvent를 발행합니다.

    Args:
        bus: EventBus (CRITICAL 시 이벤트 발행용). None이면 이벤트 발행 안 함.
    """

    def __init__(self, bus: EventBus | None = None) -> None:
        self._bus = bus
        self._checks: dict[str, HealthCheckRegistration] = {}
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[CheckResult]],
        interval: float = 60.0,
    ) -> None:
        """헬스 체크 함수 등록.

        Args:
            name: 체크 이름 (고유)
            check_fn: async 함수, CheckResult 반환
            interval: 실행 주기 (초)
        """
        self._checks[name] = HealthCheckRegistration(
            name=name,
            check_fn=check_fn,
            interval=interval,
        )

    def start(self) -> None:
        """모든 등록된 체크의 주기적 실행 시작."""
        if self._running:
            return
        self._running = True
        for reg in self._checks.values():
            task = asyncio.create_task(self._run_check_loop(reg))
            self._tasks.append(task)
        logger.info("HealthCheckEngine started: {} checks registered", len(self._checks))

    async def stop(self) -> None:
        """모든 체크 루프 중지."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("HealthCheckEngine stopped")

    async def run_all_once(self) -> list[CheckResult]:
        """모든 체크를 1회 실행 (테스트용).

        Returns:
            CheckResult 리스트
        """
        results: list[CheckResult] = []
        for reg in self._checks.values():
            result = await self._execute_check(reg)
            results.append(result)
        return results

    def get_status(self) -> dict[str, CheckResult | None]:
        """모든 체크의 마지막 결과 조회.

        Returns:
            {check_name: CheckResult | None}
        """
        return {name: reg.last_result for name, reg in self._checks.items()}

    def overall_status(self) -> CheckStatus:
        """전체 시스템 상태 (가장 심각한 상태).

        Returns:
            CRITICAL > DEGRADED > HEALTHY 우선순위
        """
        statuses = [reg.last_result.status for reg in self._checks.values() if reg.last_result]
        if not statuses:
            return CheckStatus.HEALTHY
        if CheckStatus.CRITICAL in statuses:
            return CheckStatus.CRITICAL
        if CheckStatus.DEGRADED in statuses:
            return CheckStatus.DEGRADED
        return CheckStatus.HEALTHY

    # ── Private ────────────────────────────────────────────────────

    async def _run_check_loop(self, reg: HealthCheckRegistration) -> None:
        """단일 체크의 주기적 실행 루프."""
        while self._running:
            await self._execute_check(reg)
            await asyncio.sleep(reg.interval)

    async def _execute_check(self, reg: HealthCheckRegistration) -> CheckResult:
        """단일 체크 실행 + 결과 저장 + CRITICAL 시 이벤트 발행."""
        try:
            result = await reg.check_fn()
        except Exception as exc:
            result = CheckResult(
                name=reg.name,
                status=CheckStatus.CRITICAL,
                message=f"Check raised exception: {exc}",
                timestamp=time.monotonic(),
            )

        reg.last_result = result

        if result.status == CheckStatus.CRITICAL:
            logger.critical("HealthCheck CRITICAL: {} — {}", reg.name, result.message)
            await self._publish_alert(result)
        elif result.status == CheckStatus.DEGRADED:
            logger.warning("HealthCheck DEGRADED: {} — {}", reg.name, result.message)

        return result

    async def _publish_alert(self, result: CheckResult) -> None:
        """CRITICAL 체크 결과 시 RiskAlertEvent 발행."""
        if self._bus is None:
            return
        await self._bus.publish(
            RiskAlertEvent(
                alert_level="CRITICAL",
                message=f"HealthCheck [{result.name}]: {result.message}",
                source="health_check_engine",
            )
        )


# ── Built-in Check Factories ─────────────────────────────────────


def make_bar_freshness_check(
    get_last_bar_times: Callable[[], dict[str, float]],
    stale_threshold: float = 120.0,
) -> Callable[[], Awaitable[CheckResult]]:
    """Bar freshness 체크 팩토리.

    Args:
        get_last_bar_times: {symbol: monotonic_timestamp} 반환 함수
        stale_threshold: stale 판정 임계값 (초)

    Returns:
        async check function
    """

    async def check() -> CheckResult:
        bar_times = get_last_bar_times()
        if not bar_times:
            return CheckResult(
                name="bar_freshness",
                status=CheckStatus.HEALTHY,
                message="No bars tracked yet",
                timestamp=time.monotonic(),
            )

        now = time.monotonic()
        stale_symbols = [s for s, t in bar_times.items() if now - t > stale_threshold]

        if stale_symbols:
            return CheckResult(
                name="bar_freshness",
                status=CheckStatus.DEGRADED,
                message=f"Stale bars: {', '.join(stale_symbols)}",
                timestamp=now,
            )
        return CheckResult(
            name="bar_freshness",
            status=CheckStatus.HEALTHY,
            message=f"All {len(bar_times)} symbols fresh",
            timestamp=now,
        )

    return check


def make_ws_connectivity_check(
    get_ws_connected: Callable[[], bool],
    get_last_message_time: Callable[[], float | None],
    timeout_threshold: float = 60.0,
) -> Callable[[], Awaitable[CheckResult]]:
    """WebSocket 연결 체크 팩토리.

    Args:
        get_ws_connected: WS 연결 상태 반환 함수
        get_last_message_time: 마지막 WS 메시지 시각 (monotonic) 반환 함수
        timeout_threshold: 무데이터 timeout (초)

    Returns:
        async check function
    """

    async def check() -> CheckResult:
        now = time.monotonic()
        connected = get_ws_connected()

        if not connected:
            return CheckResult(
                name="ws_connectivity",
                status=CheckStatus.CRITICAL,
                message="WebSocket disconnected",
                timestamp=now,
            )

        last_msg = get_last_message_time()
        if last_msg is not None and now - last_msg > timeout_threshold:
            return CheckResult(
                name="ws_connectivity",
                status=CheckStatus.DEGRADED,
                message=f"No WS data for {now - last_msg:.0f}s",
                timestamp=now,
            )

        return CheckResult(
            name="ws_connectivity",
            status=CheckStatus.HEALTHY,
            message="WebSocket connected",
            timestamp=now,
        )

    return check


def make_balance_consistency_check(
    get_equity: Callable[[], float],
    threshold_pct: float = 10.0,
) -> Callable[[], Awaitable[CheckResult]]:
    """잔고 급변 감지 체크 팩토리.

    Args:
        get_equity: 현재 equity 반환 함수
        threshold_pct: 급변 임계값 (%)

    Returns:
        async check function
    """
    state: dict[str, float] = {}

    async def check() -> CheckResult:
        now = time.monotonic()
        equity = get_equity()

        prev = state.get("last_equity")
        state["last_equity"] = equity

        if prev is None or prev <= 0:
            return CheckResult(
                name="balance_consistency",
                status=CheckStatus.HEALTHY,
                message=f"Equity: ${equity:,.2f}",
                timestamp=now,
            )

        change_pct = abs(equity - prev) / prev * 100.0
        if change_pct > threshold_pct:
            return CheckResult(
                name="balance_consistency",
                status=CheckStatus.DEGRADED,
                message=f"Equity changed {change_pct:.1f}% (${prev:,.0f} → ${equity:,.0f})",
                timestamp=now,
            )

        return CheckResult(
            name="balance_consistency",
            status=CheckStatus.HEALTHY,
            message=f"Equity stable: ${equity:,.2f}",
            timestamp=now,
        )

    return check
