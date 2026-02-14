"""StrategyOrchestrator — 멀티 전략 오케스트레이터.

여러 StrategyPod의 시그널을 수집·넷팅하여 SignalEvent를 발행하고,
FillEvent를 비례 귀속합니다.

EventBus 이벤트 흐름:
    BarEvent → Orchestrator._on_bar() → Pod.compute_signal()
    → net weight 계산 → SignalEvent 발행 → PM
    FillEvent → Orchestrator._on_fill() → 비례 귀속

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - #23 Exception Handling: Pod 에러 격리
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import (
    AnyEvent,
    BarEvent,
    EventType,
    FillEvent,
    SignalEvent,
)
from src.models.types import Direction
from src.orchestrator.netting import attribute_fill, scale_weights_to_leverage

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd

    from src.core.event_bus import EventBus
    from src.notification.orchestrator_engine import OrchestratorNotificationEngine
    from src.orchestrator.allocator import CapitalAllocator
    from src.orchestrator.config import OrchestratorConfig
    from src.orchestrator.lifecycle import LifecycleManager
    from src.orchestrator.models import RiskAlert
    from src.orchestrator.pod import StrategyPod
    from src.orchestrator.risk_aggregator import RiskAggregator

# ── Constants ─────────────────────────────────────────────────────

_ORCHESTRATOR_SOURCE = "StrategyOrchestrator"
_MIN_NET_WEIGHT = 1e-8
_RISK_DEFENSE_SCALE = 0.5
_MIN_FRACTION_EPSILON = 1e-12


# ── StrategyOrchestrator ─────────────────────────────────────────


class StrategyOrchestrator:
    """멀티 전략 오케스트레이터.

    여러 StrategyPod의 시그널을 수집·넷팅하여 SignalEvent를 발행하고,
    FillEvent를 비례 귀속합니다.

    Args:
        config: OrchestratorConfig
        pods: StrategyPod 리스트
        allocator: CapitalAllocator 인스턴스
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        pods: list[StrategyPod],
        allocator: CapitalAllocator,
        lifecycle_manager: LifecycleManager | None = None,
        risk_aggregator: RiskAggregator | None = None,
        target_timeframe: str | None = None,
        notification: OrchestratorNotificationEngine | None = None,
    ) -> None:
        self._config = config
        self._pods = pods
        self._allocator = allocator
        self._lifecycle = lifecycle_manager
        self._risk_aggregator = risk_aggregator
        self._target_timeframe = target_timeframe
        self._notification: OrchestratorNotificationEngine | None = notification
        self._bus: EventBus | None = None

        # 심볼 → Pod 인덱스 라우팅 테이블 (O(1) lookup)
        self._symbol_pod_map: dict[str, list[int]] = {}
        self._build_routing_table()

        # pod_id → Pod O(1) lookup
        self._pod_map: dict[str, StrategyPod] = {p.pod_id: p for p in pods}

        # 배치 시그널
        self._pending_bar_ts: datetime | None = None
        self._pending_net_weights: dict[str, float] = {}

        # Fill attribution용: pod_id → {symbol → global_target}
        self._last_pod_targets: dict[str, dict[str, float]] = {}

        # 리밸런스 추적
        self._last_rebalance_ts: datetime | None = None
        self._last_allocated_weights: dict[str, float] = {}

        # Fire-and-forget notification tasks (prevent GC)
        self._notification_tasks: set[asyncio.Task[None]] = set()

        # Risk defense
        self._risk_breached: bool = False

        # History 추적
        self._allocation_history: list[dict[str, object]] = []
        self._lifecycle_events: list[dict[str, object]] = []
        self._risk_contributions_history: list[dict[str, object]] = []

    def set_notification_engine(self, engine: OrchestratorNotificationEngine) -> None:
        """알림 엔진 주입 (LiveRunner에서 후에 설정)."""
        self._notification = engine

    def _fire_notification(self, coro: object) -> None:
        """Fire-and-forget notification coroutine. 참조를 보관하여 GC 방지."""
        task = asyncio.create_task(coro)  # type: ignore[arg-type]
        self._notification_tasks.add(task)
        task.add_done_callback(self._notification_tasks.discard)

    # ── EventBus Integration ──────────────────────────────────────

    async def register(self, bus: EventBus) -> None:
        """EventBus에 BAR, FILL 구독 등록."""
        self._bus = bus
        bus.subscribe(EventType.BAR, self._on_bar)
        bus.subscribe(EventType.FILL, self._on_fill)

    # ── Event Handlers ───────────────────────────────────────────

    async def _on_bar(self, event: AnyEvent) -> None:
        """BarEvent 핸들러: Pod 라우팅 → 시그널 수집 → 넷팅."""
        assert isinstance(event, BarEvent)
        bar = event

        # TF 필터: target_timeframe이 설정된 경우, 해당 TF bar만 처리
        if self._target_timeframe is not None and bar.timeframe != self._target_timeframe:
            return

        symbol = bar.symbol

        # 1. 새 bar_timestamp → 이전 배치 flush
        if self._pending_bar_ts is not None and bar.bar_timestamp != self._pending_bar_ts:
            await self._flush_net_signals()

        self._pending_bar_ts = bar.bar_timestamp

        # 2. 심볼 → Pod 라우팅
        pod_indices = self._symbol_pod_map.get(symbol, [])
        if not pod_indices:
            return

        # 3. 각 active Pod에 시그널 계산
        bar_data = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }

        for idx in pod_indices:
            pod = self._pods[idx]
            if not pod.is_active:
                continue

            result = pod.compute_signal(symbol, bar_data, bar.bar_timestamp)
            if result is None:
                continue

            direction, strength = result
            # global_weight = direction * strength * capital_fraction
            global_weight = direction * strength * pod.capital_fraction

            # last_pod_targets에 저장 (fill attribution용)
            if pod.pod_id not in self._last_pod_targets:
                self._last_pod_targets[pod.pod_id] = {}
            self._last_pod_targets[pod.pod_id][symbol] = global_weight

            # 심볼별 net weight 누적
            current = self._pending_net_weights.get(symbol, 0.0)
            self._pending_net_weights[symbol] = current + global_weight

        # 4. 리밸런스 체크
        self._check_rebalance(bar.bar_timestamp)

    async def _on_fill(self, event: AnyEvent) -> None:
        """FillEvent 핸들러: netting.attribute_fill()로 비례 귀속."""
        assert isinstance(event, FillEvent)
        fill = event

        symbol = fill.symbol
        is_buy = fill.side == "BUY"

        # 심볼에 대한 Pod별 타겟 수집
        pod_targets: dict[str, float] = {}
        for pod_id, targets in self._last_pod_targets.items():
            if symbol in targets:
                pod_targets[pod_id] = targets[symbol]

        if not pod_targets:
            return

        attributed = attribute_fill(
            symbol,
            fill.fill_qty,
            fill.fill_price,
            fill.fee,
            pod_targets,
            is_buy=is_buy,
        )

        for pod_id, (attr_qty, attr_price, attr_fee) in attributed.items():
            pod = self._find_pod(pod_id)
            if pod is not None:
                pod.update_position(
                    symbol,
                    attr_qty,
                    attr_price,
                    attr_fee,
                    is_buy=is_buy,
                )

    # ── Signal Emission ──────────────────────────────────────────

    async def _flush_net_signals(self) -> None:
        """누적된 net weight → SignalEvent 발행."""
        bus = self._bus
        if bus is None or self._pending_bar_ts is None:
            return

        # Risk breached → 모든 weight 0으로 억제
        if self._risk_breached:
            self._pending_net_weights = dict.fromkeys(self._pending_net_weights, 0.0)

        # 레버리지 한도 적용
        scaled = scale_weights_to_leverage(
            self._pending_net_weights,
            self._config.max_gross_leverage,
        )

        for symbol, net_weight in scaled.items():
            abs_weight = abs(net_weight)
            if abs_weight < _MIN_NET_WEIGHT:
                direction = Direction.NEUTRAL
                strength = 0.0
            elif net_weight > 0:
                direction = Direction.LONG
                strength = abs_weight
            else:
                direction = Direction.SHORT
                strength = abs_weight

            signal = SignalEvent(
                symbol=symbol,
                strategy_name=_ORCHESTRATOR_SOURCE,
                direction=direction,
                strength=strength,
                bar_timestamp=self._pending_bar_ts,
                source=_ORCHESTRATOR_SOURCE,
            )
            await bus.publish(signal)

        self._pending_net_weights.clear()
        self._pending_bar_ts = None

    async def flush_pending_signals(self) -> None:
        """외부(Runner)에서 호출: 누적 시그널 flush."""
        await self._flush_net_signals()

    # ── Rebalance ────────────────────────────────────────────────

    def _check_rebalance(self, bar_timestamp: datetime) -> None:
        """Calendar/Threshold/Hybrid 리밸런스 트리거 확인."""
        trigger = self._config.rebalance_trigger

        calendar_due = False
        if self._last_rebalance_ts is None:
            calendar_due = True
        else:
            days_since = (bar_timestamp - self._last_rebalance_ts).days
            calendar_due = days_since >= self._config.rebalance_calendar_days

        threshold_due = self._check_drift_threshold()

        should_rebalance = False
        if trigger.value == "calendar":
            should_rebalance = calendar_due
        elif trigger.value == "threshold":
            should_rebalance = threshold_due
        elif trigger.value == "hybrid":
            should_rebalance = calendar_due or threshold_due

        if should_rebalance:
            self._execute_rebalance()
            self._last_rebalance_ts = bar_timestamp

    def _check_drift_threshold(self) -> bool:
        """현재 capital_fraction과 last allocated weight 비교하여 drift 확인."""
        for pod in self._pods:
            if not pod.is_active:
                continue
            target = self._last_allocated_weights.get(
                pod.pod_id, pod.config.initial_fraction
            )
            drift = abs(pod.capital_fraction - target)
            if drift > self._config.rebalance_drift_threshold:
                return True
        return False

    def _execute_rebalance(self) -> None:
        """Allocator.compute_weights() 호출 → Pod capital_fraction 업데이트."""
        import pandas as pd

        # Lifecycle 평가 (weight 계산 전에 상태 전이)
        self._evaluate_lifecycle()

        # Pod 수익률 DataFrame 구성
        pod_returns_data: dict[str, list[float]] = {}
        for pod in self._pods:
            if pod.is_active:
                returns = list(pod.daily_returns_series)
                pod_returns_data[pod.pod_id] = returns if returns else [0.0]

        if not pod_returns_data:
            return

        max_len = max(len(v) for v in pod_returns_data.values())
        for pid, current in pod_returns_data.items():
            if len(current) < max_len:
                pod_returns_data[pid] = [0.0] * (max_len - len(current)) + current

        pod_returns = pd.DataFrame(pod_returns_data)

        pod_states = {pod.pod_id: pod.state for pod in self._pods}
        pod_live_days = {pod.pod_id: pod.performance.live_days for pod in self._pods}

        new_weights = self._allocator.compute_weights(
            pod_returns,
            pod_states,
            lookback=self._config.correlation_lookback,
            pod_live_days=pod_live_days,
        )

        for pod in self._pods:
            if pod.pod_id in new_weights:
                pod.capital_fraction = new_weights[pod.pod_id]

        self._last_allocated_weights = dict(new_weights)

        logger.debug("Rebalanced: {}", {p.pod_id: p.capital_fraction for p in self._pods})

        # Allocation history 기록
        alloc_record: dict[str, object] = {"timestamp": self._pending_bar_ts}
        for pod in self._pods:
            alloc_record[pod.pod_id] = pod.capital_fraction
        self._allocation_history.append(alloc_record)

        # Rebalance 알림
        if self._notification is not None:
            allocations = {p.pod_id: p.capital_fraction for p in self._pods}
            self._fire_notification(
                self._notification.notify_capital_rebalance(
                    timestamp=str(self._pending_bar_ts),
                    allocations=allocations,
                    trigger_reason=self._config.rebalance_trigger.value,
                )
            )

        # Risk 한도 검사 + history 기록
        net_weights = self._compute_current_net_weights()
        self._check_risk_limits(pod_returns, net_weights=net_weights)

    def _evaluate_lifecycle(self) -> None:
        """Lifecycle 평가 → 상태 전이 감지 → lifecycle_events 기록."""
        if self._lifecycle is None:
            return

        portfolio_returns_series = self._compute_portfolio_returns()
        for pod in self._pods:
            if not pod.is_active:
                continue
            pre_state = pod.state.value
            self._lifecycle.evaluate(
                pod, portfolio_returns_series, bar_timestamp=self._pending_bar_ts
            )
            if pod.state.value != pre_state:
                self._lifecycle_events.append(
                    {
                        "pod_id": pod.pod_id,
                        "from_state": pre_state,
                        "to_state": pod.state.value,
                        "timestamp": self._pending_bar_ts,
                    }
                )
                if self._notification is not None:
                    self._fire_notification(
                        self._notification.notify_lifecycle_transition(
                            pod_id=pod.pod_id,
                            from_state=pre_state,
                            to_state=pod.state.value,
                            timestamp=str(self._pending_bar_ts),
                        )
                    )

    def _compute_daily_pnl_pct(self) -> float:
        """활성 Pod의 가중 평균 최신 일간 수익률."""
        active_pods = [p for p in self._pods if p.is_active and len(p.daily_returns) > 0]
        if not active_pods:
            return 0.0
        total_fraction = sum(p.capital_fraction for p in active_pods)
        if total_fraction < _MIN_FRACTION_EPSILON:
            return 0.0
        weighted = sum(p.daily_returns[-1] * p.capital_fraction for p in active_pods)
        return weighted / total_fraction

    def _compute_current_net_weights(self) -> dict[str, float]:
        """last_pod_targets에서 현재 net weights 재구성."""
        net: dict[str, float] = {}
        for targets in self._last_pod_targets.values():
            for symbol, weight in targets.items():
                net[symbol] = net.get(symbol, 0.0) + weight
        return net

    def _check_risk_limits(
        self,
        pod_returns: pd.DataFrame,
        *,
        net_weights: dict[str, float] | None = None,
    ) -> list[RiskAlert]:
        """Risk 한도 검사 + risk contributions history 기록."""
        if self._risk_aggregator is None:
            return []

        weights_to_check = net_weights if net_weights is not None else self._pending_net_weights

        pod_performances = {pod.pod_id: pod.performance for pod in self._pods if pod.is_active}
        pod_weights_map = {pod.pod_id: pod.capital_fraction for pod in self._pods if pod.is_active}
        daily_pnl = self._compute_daily_pnl_pct()
        alerts = self._risk_aggregator.check_portfolio_limits(
            net_weights=weights_to_check,
            pod_performances=pod_performances,
            pod_weights=pod_weights_map,
            pod_returns=pod_returns,
            daily_pnl_pct=daily_pnl,
        )
        for alert in alerts:
            logger.warning("RiskAlert [{}]: {}", alert.severity, alert.message)

        if alerts and self._notification is not None:
            self._fire_notification(self._notification.notify_risk_alerts(alerts))

        # Risk contributions history 기록
        risk_record: dict[str, object] = {"timestamp": self._pending_bar_ts}
        risk_record.update(pod_weights_map)
        self._risk_contributions_history.append(risk_record)

        # Risk defense: critical alert → capital 축소
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        if critical_alerts:
            self._apply_risk_defense(critical_alerts)
        elif self._risk_breached:
            self._risk_breached = False
            logger.info("Risk defense deactivated — no critical alerts")

        return alerts

    def _apply_risk_defense(self, alerts: list[RiskAlert]) -> None:
        """CRITICAL alert 시 활성 Pod의 capital_fraction을 축소."""
        self._risk_breached = True
        for pod in self._pods:
            if pod.is_active:
                pod.capital_fraction *= _RISK_DEFENSE_SCALE
        logger.warning(
            "Risk defense: {} critical alerts, capital scaled by {:.0%}",
            len(alerts),
            _RISK_DEFENSE_SCALE,
        )

    # ── Query ────────────────────────────────────────────────────

    def get_pod_summary(self) -> list[dict[str, object]]:
        """Pod 요약 정보 리스트."""
        return [
            {
                "pod_id": pod.pod_id,
                "state": pod.state.value,
                "capital_fraction": pod.capital_fraction,
                "is_active": pod.is_active,
                "trade_count": pod.performance.trade_count,
                "live_days": pod.performance.live_days,
            }
            for pod in self._pods
        ]

    @property
    def active_pod_count(self) -> int:
        """활성 Pod 수."""
        return sum(1 for pod in self._pods if pod.is_active)

    @property
    def pods(self) -> list[StrategyPod]:
        """Pod 리스트."""
        return list(self._pods)

    @property
    def lifecycle(self) -> LifecycleManager | None:
        """Lifecycle 매니저."""
        return self._lifecycle

    @property
    def allocation_history(self) -> list[dict[str, object]]:
        """자본 배분 이력."""
        return self._allocation_history

    @property
    def lifecycle_events(self) -> list[dict[str, object]]:
        """생애주기 상태 전이 이력."""
        return self._lifecycle_events

    @property
    def risk_contributions_history(self) -> list[dict[str, object]]:
        """리스크 기여도 이력."""
        return self._risk_contributions_history

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Serialize orchestrator-level state for persistence.

        Excludes in-memory histories (_allocation_history, _lifecycle_events,
        _risk_contributions_history) — deferred to Phase 11.
        """
        return {
            "last_rebalance_ts": (
                self._last_rebalance_ts.isoformat() if self._last_rebalance_ts is not None else None
            ),
            "last_pod_targets": {
                pod_id: dict(targets) for pod_id, targets in self._last_pod_targets.items()
            },
        }

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Restore orchestrator-level state from persisted dict."""
        from datetime import datetime

        ts_val = data.get("last_rebalance_ts")
        if isinstance(ts_val, str):
            self._last_rebalance_ts = datetime.fromisoformat(ts_val)
        else:
            self._last_rebalance_ts = None

        targets_val = data.get("last_pod_targets")
        if isinstance(targets_val, dict):
            self._last_pod_targets = {
                str(pod_id): {str(sym): float(w) for sym, w in syms.items()}
                for pod_id, syms in targets_val.items()
                if isinstance(syms, dict)
            }

    # ── Private ──────────────────────────────────────────────────

    def _compute_portfolio_returns(self) -> pd.Series | None:
        """활성 Pod의 가중 평균 일별 수익률 계산."""
        import pandas as pd

        active_pods = [p for p in self._pods if p.is_active and len(p.daily_returns) > 0]
        if not active_pods:
            return None

        total_fraction = sum(p.capital_fraction for p in active_pods)
        if total_fraction == 0.0:
            return None

        max_len = max(len(p.daily_returns) for p in active_pods)
        weighted_sum = pd.Series([0.0] * max_len, dtype=float)

        for pod in active_pods:
            returns = pod.daily_returns
            padded = [0.0] * (max_len - len(returns)) + list(returns)
            weight = pod.capital_fraction / total_fraction
            weighted_sum += pd.Series(padded, dtype=float) * weight

        return weighted_sum

    def _build_routing_table(self) -> None:
        """심볼 → Pod 인덱스 라우팅 테이블 구축."""
        self._symbol_pod_map.clear()
        for idx, pod in enumerate(self._pods):
            for symbol in pod.symbols:
                if symbol not in self._symbol_pod_map:
                    self._symbol_pod_map[symbol] = []
                self._symbol_pod_map[symbol].append(idx)

    def _find_pod(self, pod_id: str) -> StrategyPod | None:
        """pod_id로 Pod 조회 (O(1))."""
        return self._pod_map.get(pod_id)
