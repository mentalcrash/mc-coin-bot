"""Portfolio Risk Monitor -- 포트폴리오 수준 실시간 리스크 감시.

EventBus subscriber로 BALANCE_UPDATE, FILL, BAR 이벤트를 구독하여
포트폴리오 MDD, 일일 손실, 상관 집중도, 단일 심볼 집중도를 모니터링합니다.

임계값 초과 시 RiskAlertEvent/CircuitBreakerEvent를 발행합니다.

Rules Applied:
    - MetricsExporter EventBus 등록 패턴
    - risk_aggregator.py pure functions 재사용
    - RiskAlertEvent/CircuitBreakerEvent 기존 이벤트 재사용
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    BarEvent,
    CircuitBreakerEvent,
    EventType,
    FillEvent,
    RiskAlertEvent,
)
from src.orchestrator.risk_aggregator import check_asset_correlation_stress
from src.portfolio.risk_monitor_models import (
    PortfolioRiskConfig,
    PortfolioRiskSnapshot,
    RiskAction,
)

if TYPE_CHECKING:
    from src.core.event_bus import EventBus

# ── Constants ─────────────────────────────────────────────────────
_WARNING_RATIO = 0.8
_MIN_PRICE_HISTORY = 5  # 상관 계산 최소 데이터 포인트


class PortfolioRiskMonitor:
    """포트폴리오 수준 실시간 리스크 모니터.

    EventBus에 등록하여 BALANCE_UPDATE, FILL, BAR 이벤트를 구독하고,
    리스크 임계값 초과 시 RiskAlertEvent/CircuitBreakerEvent를 발행합니다.

    Args:
        config: 리스크 설정 (기본값 사용 가능)
    """

    def __init__(self, config: PortfolioRiskConfig | None = None) -> None:
        self._config = config or PortfolioRiskConfig()
        self._bus: EventBus | None = None

        # State tracking
        self._initial_equity: float = 0.0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._day_start_equity: float = 0.0
        self._current_day: int = -1  # day of year

        # Per-symbol tracking
        self._symbol_notional: dict[str, float] = {}
        self._price_history: dict[str, list[float]] = {}

        # Last snapshot
        self._last_snapshot: PortfolioRiskSnapshot | None = None

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록.

        Args:
            bus: EventBus 인스턴스
        """
        self._bus = bus
        bus.subscribe(EventType.BALANCE_UPDATE, self._on_balance_update)
        bus.subscribe(EventType.FILL, self._on_fill)
        bus.subscribe(EventType.BAR, self._on_bar)
        logger.info("PortfolioRiskMonitor registered")

    def set_initial_equity(self, equity: float) -> None:
        """초기 equity 설정 (LiveRunner startup 시).

        Args:
            equity: 초기 자본
        """
        self._initial_equity = equity
        self._peak_equity = equity
        self._current_equity = equity
        self._day_start_equity = equity

    @property
    def last_snapshot(self) -> PortfolioRiskSnapshot | None:
        """마지막 리스크 스냅샷."""
        return self._last_snapshot

    # ── Event handlers ─────────────────────────────────────────────

    async def _on_balance_update(self, event: AnyEvent) -> None:
        """BALANCE_UPDATE 이벤트 처리."""
        assert isinstance(event, BalanceUpdateEvent)

        self._current_equity = event.total_equity
        self._peak_equity = max(self._peak_equity, event.total_equity)

        # 일일 시작 equity 업데이트
        now = datetime.now(UTC)
        day_of_year = now.timetuple().tm_yday
        if day_of_year != self._current_day:
            self._current_day = day_of_year
            self._day_start_equity = self._current_equity

        await self._evaluate_risk()

    async def _on_fill(self, event: AnyEvent) -> None:
        """FILL 이벤트 처리 — 심볼별 notional 추적."""
        assert isinstance(event, FillEvent)

        notional = event.fill_price * event.fill_qty
        if event.side == "BUY":
            self._symbol_notional[event.symbol] = (
                self._symbol_notional.get(event.symbol, 0.0) + notional
            )
        else:
            self._symbol_notional[event.symbol] = (
                self._symbol_notional.get(event.symbol, 0.0) - notional
            )

    async def _on_bar(self, event: AnyEvent) -> None:
        """BAR 이벤트 처리 — 가격 히스토리 수집."""
        assert isinstance(event, BarEvent)

        history = self._price_history.setdefault(event.symbol, [])
        history.append(event.close)

        # 메모리 관리: 최근 100개만 유지
        _max_history = 100
        if len(history) > _max_history:
            self._price_history[event.symbol] = history[-_max_history:]

    # ── Risk evaluation ────────────────────────────────────────────

    async def _evaluate_risk(self) -> None:
        """현재 상태 기반 리스크 평가 + 이벤트 발행."""
        action = RiskAction.NORMAL
        messages: list[str] = []

        dd = self._compute_drawdown()
        daily_pnl = self._compute_daily_pnl()
        concentration = self._compute_max_concentration()
        avg_corr = self._compute_avg_correlation()

        # 1) Portfolio drawdown check
        action, messages = self._check_drawdown(dd, action, messages)

        # 2) Daily loss check
        action, messages = self._check_daily_loss(daily_pnl, action, messages)

        # 3) Concentration check
        action, messages = self._check_concentration(concentration, action, messages)

        # 4) Correlation check
        action, messages = self._check_correlation(avg_corr, action, messages)

        # Snapshot 생성
        self._last_snapshot = PortfolioRiskSnapshot(
            timestamp=datetime.now(UTC),
            action=action,
            portfolio_drawdown=dd,
            daily_pnl_pct=daily_pnl,
            max_concentration=concentration,
            avg_correlation=avg_corr,
            messages=messages,
        )

        # 이벤트 발행
        await self._publish_alerts(action, messages)

    def _check_drawdown(
        self, dd: float, action: RiskAction, messages: list[str]
    ) -> tuple[RiskAction, list[str]]:
        """Drawdown 검사."""
        threshold = self._config.max_portfolio_drawdown
        if dd >= threshold:
            action = RiskAction.LIQUIDATE_ALL
            messages = [*messages, f"Portfolio drawdown {dd:.2%} >= limit {threshold:.1%}"]
        elif dd >= threshold * _WARNING_RATIO:
            messages = [*messages, f"Portfolio drawdown {dd:.2%} approaching limit {threshold:.1%}"]
        return action, messages

    def _check_daily_loss(
        self, daily_pnl: float, action: RiskAction, messages: list[str]
    ) -> tuple[RiskAction, list[str]]:
        """일일 손실 검사."""
        threshold = self._config.max_daily_loss
        loss = abs(min(daily_pnl, 0.0))
        if loss >= threshold and action != RiskAction.LIQUIDATE_ALL:
            action = RiskAction.HALT_NEW_ENTRIES
            messages = [*messages, f"Daily loss {loss:.2%} >= limit {threshold:.1%}"]
        elif loss >= threshold * _WARNING_RATIO:
            messages = [*messages, f"Daily loss {loss:.2%} approaching limit {threshold:.1%}"]
        return action, messages

    def _check_concentration(
        self, concentration: float, action: RiskAction, messages: list[str]
    ) -> tuple[RiskAction, list[str]]:
        """집중도 검사."""
        threshold = self._config.max_concentration_pct
        if concentration >= threshold:
            messages = [
                *messages,
                f"Concentration {concentration:.2%} >= limit {threshold:.1%}",
            ]
        return action, messages

    def _check_correlation(
        self, avg_corr: float, action: RiskAction, messages: list[str]
    ) -> tuple[RiskAction, list[str]]:
        """상관 검사."""
        threshold = self._config.max_correlation_exposure
        if avg_corr >= threshold and action not in (
            RiskAction.LIQUIDATE_ALL,
            RiskAction.HALT_NEW_ENTRIES,
        ):
            action = RiskAction.REDUCE_CORRELATED
            messages = [
                *messages,
                f"Avg correlation {avg_corr:.2f} >= limit {threshold:.2f}",
            ]
        elif avg_corr >= threshold * _WARNING_RATIO:
            messages = [
                *messages,
                f"Avg correlation {avg_corr:.2f} approaching limit {threshold:.2f}",
            ]
        return action, messages

    # ── Metric helpers ─────────────────────────────────────────────

    def _compute_drawdown(self) -> float:
        """현재 HWM 대비 낙폭 계산."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._current_equity) / self._peak_equity)

    def _compute_daily_pnl(self) -> float:
        """당일 PnL 비율 계산."""
        if self._day_start_equity <= 0:
            return 0.0
        return (self._current_equity - self._day_start_equity) / self._day_start_equity

    def _compute_max_concentration(self) -> float:
        """단일 심볼 최대 비중 계산."""
        if self._current_equity <= 0 or not self._symbol_notional:
            return 0.0
        total_notional = sum(abs(v) for v in self._symbol_notional.values())
        if total_notional <= 0:
            return 0.0
        max_notional = max(abs(v) for v in self._symbol_notional.values())
        return max_notional / total_notional

    def _compute_avg_correlation(self) -> float:
        """에셋 간 평균 상관계수 계산 (risk_aggregator 재사용)."""
        # 최소 데이터 포인트 확보된 심볼만 사용
        eligible = {
            s: prices
            for s, prices in self._price_history.items()
            if len(prices) >= _MIN_PRICE_HISTORY
        }
        _min_assets_for_corr = 2
        if len(eligible) < _min_assets_for_corr:
            return 0.0

        _, avg_corr = check_asset_correlation_stress(eligible, threshold=2.0)
        return avg_corr

    # ── Alert publishing ───────────────────────────────────────────

    async def _publish_alerts(self, action: RiskAction, messages: list[str]) -> None:
        """리스크 액션에 따라 이벤트 발행."""
        if self._bus is None or not messages:
            return

        if action == RiskAction.LIQUIDATE_ALL:
            await self._bus.publish(
                CircuitBreakerEvent(
                    reason=f"PortfolioRiskMonitor: {'; '.join(messages)}",
                    close_all_positions=True,
                    source="portfolio_risk_monitor",
                )
            )
            logger.critical("PortfolioRiskMonitor: LIQUIDATE_ALL — {}", "; ".join(messages))
        elif action in (RiskAction.HALT_NEW_ENTRIES, RiskAction.REDUCE_CORRELATED):
            await self._bus.publish(
                RiskAlertEvent(
                    alert_level="CRITICAL",
                    message=f"PortfolioRiskMonitor [{action.value}]: {'; '.join(messages)}",
                    source="portfolio_risk_monitor",
                )
            )
            logger.warning("PortfolioRiskMonitor: {} — {}", action.value, "; ".join(messages))
        else:
            # WARNING level messages
            for msg in messages:
                await self._bus.publish(
                    RiskAlertEvent(
                        alert_level="WARNING",
                        message=f"PortfolioRiskMonitor: {msg}",
                        source="portfolio_risk_monitor",
                    )
                )
