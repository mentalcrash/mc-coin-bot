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
from datetime import date
from typing import TYPE_CHECKING

from loguru import logger

from src.core.events import (
    AnyEvent,
    BarEvent,
    EventType,
    FillEvent,
    RiskAlertEvent,
    SignalEvent,
)
from src.models.types import Direction
from src.orchestrator.netting import (
    attribute_fill,
    compute_netting_stats,
    scale_weights_to_leverage,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
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
    from src.orchestrator.surveillance import ScanResult

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

        # Per-pod TF routing: accepted TF set
        if target_timeframe is not None:
            self._accepted_timeframes: set[str] = {target_timeframe}
        else:
            self._accepted_timeframes = {pod.config.timeframe for pod in pods}

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
        self._risk_recovery_step: int = 0  # 0=정상, 1~N=복원 중

        # Daily return MTM 추적
        self._initial_capital: float = 0.0
        self._last_day_date: date | None = None
        self._last_close_prices: dict[str, float] = {}
        self._pending_day_change: bool = False
        self._day_end_close_prices: dict[str, float] = {}  # day boundary snapshot

        # Asset close price history for correlation (Pod < 3 보완)
        self._close_price_history: dict[str, list[float]] = {}

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

        # TF 필터: accepted TF만 처리
        if bar.timeframe not in self._accepted_timeframes:
            return

        symbol = bar.symbol

        # Day boundary detection — MTM은 signal flush 시점으로 지연
        # (모든 심볼의 close price가 갱신된 후 기록)
        bar_date = bar.bar_timestamp.date()
        if self._last_day_date is not None and bar_date != self._last_day_date:
            self._pending_day_change = True
            # Snapshot: 전일 모든 심볼의 close price (day boundary 직전 상태)
            self._day_end_close_prices = dict(self._last_close_prices)
        self._last_day_date = bar_date

        # Close price 저장 (snapshot 이후 갱신)
        self._last_close_prices[symbol] = bar.close

        # Close price history for asset-level correlation
        if symbol not in self._close_price_history:
            self._close_price_history[symbol] = []
        self._close_price_history[symbol].append(bar.close)
        lookback = self._config.correlation_lookback
        if len(self._close_price_history[symbol]) > lookback:
            self._close_price_history[symbol] = self._close_price_history[symbol][-lookback:]

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
            if pod.config.timeframe != bar.timeframe:
                continue

            result = pod.compute_signal(symbol, bar_data, bar.bar_timestamp)
            if result is None:
                continue

            # E2: all-excluded pod → 시그널 누적 안 함 (내부 상태만 갱신)
            if not pod.should_emit_signals:
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
        """누적된 net weight → SignalEvent 발행.

        Multi-TF 모드: _last_pod_targets에서 전체 net weight를 재계산하되,
        이번 배치에서 변경된 심볼만 emit하여 불필요한 주문을 방지합니다.
        """
        bus = self._bus
        if bus is None or self._pending_bar_ts is None:
            return

        # Day change MTM: 전일 close price snapshot으로 기록
        if self._pending_day_change:
            self._record_pod_daily_returns(self._day_end_close_prices)
            self._pending_day_change = False

        # 이번 배치에서 변경된 심볼
        changed_symbols = set(self._pending_net_weights.keys())

        # Layer 2: Per-pod aggregate leverage cap
        self._apply_per_pod_leverage_cap()

        # full_net 계산 (per-pod cap 반영)
        # Multi-TF / 단일 TF 모두 _last_pod_targets 기반으로 통일
        full_net = self._compute_current_net_weights()

        # Risk breached → 모든 weight 0으로 억제
        if self._risk_breached:
            full_net = dict.fromkeys(full_net, 0.0)
        elif self._risk_recovery_step > 0:
            # 점진 복구: step/total_steps 비율로 weight 스케일링
            fraction = self._risk_recovery_step / self._config.risk_recovery_steps
            full_net = {sym: w * fraction for sym, w in full_net.items()}

        # Netting 상쇄 모니터링
        if self._last_pod_targets:
            netting_stats = compute_netting_stats(self._last_pod_targets)
            threshold = self._config.netting_offset_warning_threshold
            if netting_stats.offset_ratio > threshold:
                msg = (
                    f"High netting offset: {netting_stats.offset_ratio:.1%} "
                    f"(gross={netting_stats.gross_sum:.4f}, net={netting_stats.net_sum:.4f})"
                )
                logger.warning(msg)
                alert = RiskAlertEvent(
                    alert_level="WARNING",
                    message=msg,
                    source=_ORCHESTRATOR_SOURCE,
                )
                await bus.publish(alert)

        # 레버리지 한도 적용
        scaled = scale_weights_to_leverage(
            full_net,
            self._config.max_gross_leverage,
        )

        for symbol in changed_symbols:
            net_weight = scaled.get(symbol, 0.0)
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
            target = self._last_allocated_weights.get(pod.pod_id, pod.config.initial_fraction)
            drift = abs(pod.capital_fraction - target)
            if drift > self._config.rebalance_drift_threshold:
                return True
        return False

    def _execute_rebalance(self) -> None:
        """Allocator.compute_weights() 호출 → Pod capital_fraction 업데이트."""
        import pandas as pd

        # Lifecycle 평가 (weight 계산 전에 상태 전이)
        self._evaluate_lifecycle()

        # E7: Retired pod 잔여 타겟 즉시 zero-out
        self._cleanup_retired_pod_targets()

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

        # Apply turnover constraints (clamp per-pod and total delta)
        new_weights = self._apply_turnover_constraints(new_weights)

        # Turnover filter: skip if total turnover is below threshold
        total_turnover = sum(
            abs(new_weights.get(pod.pod_id, 0.0) - pod.capital_fraction)
            for pod in self._pods
            if pod.is_active
        )
        if total_turnover < self._config.min_rebalance_turnover:
            logger.debug(
                "Rebalance skipped: turnover {:.4f} < min {:.4f}",
                total_turnover,
                self._config.min_rebalance_turnover,
            )
            return

        for pod in self._pods:
            if pod.pod_id in new_weights:
                pod.capital_fraction = new_weights[pod.pod_id]
                # Equity 연속성 보장: 리밸런스 후 spike 방지
                if self._initial_capital > 0:
                    pod_prices = {
                        s: self._last_close_prices[s]
                        for s in pod.symbols
                        if s in self._last_close_prices
                    }
                    pod.adjust_base_equity_on_rebalance(
                        self._initial_capital * new_weights[pod.pod_id],
                        pod_prices,
                    )

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

    def _apply_turnover_constraints(self, new_weights: dict[str, float]) -> dict[str, float]:
        """Pod별/전체 턴오버를 제약합니다.

        1. RETIRED Pod은 bypass (즉시 0)
        2. Pod별 |delta| <= max_pod_turnover_per_rebalance 클램프
        3. 전체 sum(|delta|) <= max_total_turnover_per_rebalance 초과 시 비례 축소

        Args:
            new_weights: allocator가 산출한 새 가중치 {pod_id: fraction}

        Returns:
            제약 적용된 가중치 {pod_id: fraction}
        """
        from src.orchestrator.models import LifecycleState

        tc = self._config.turnover_constraint
        if tc is None:
            return new_weights

        constrained: dict[str, float] = {}
        for pod in self._pods:
            pid = pod.pod_id
            target = new_weights.get(pid)
            if target is None:
                continue

            # RETIRED → bypass (즉시 0)
            if pod.state == LifecycleState.RETIRED:
                constrained[pid] = target
                continue

            # Per-pod delta clamp
            current = pod.capital_fraction
            delta = target - current
            max_delta = tc.max_pod_turnover_per_rebalance
            clamped_delta = max(-max_delta, min(delta, max_delta))
            constrained[pid] = current + clamped_delta

        # Total turnover check
        total_abs_delta = sum(
            abs(constrained.get(pod.pod_id, pod.capital_fraction) - pod.capital_fraction)
            for pod in self._pods
            if pod.state != LifecycleState.RETIRED and pod.pod_id in constrained
        )

        max_total = tc.max_total_turnover_per_rebalance
        if total_abs_delta > max_total and total_abs_delta > _MIN_FRACTION_EPSILON:
            scale = max_total / total_abs_delta
            for pod in self._pods:
                pid = pod.pod_id
                if pid not in constrained or pod.state == LifecycleState.RETIRED:
                    continue
                current = pod.capital_fraction
                delta = constrained[pid] - current
                constrained[pid] = current + delta * scale

        return constrained

    def _evaluate_lifecycle(self) -> None:
        """Lifecycle 평가 → 상태 전이 감지 → lifecycle_events 기록 → GBM alerts."""
        if self._lifecycle is None:
            return

        from src.monitoring.anomaly.gbm_drawdown import DrawdownSeverity

        portfolio_returns_series = self._compute_portfolio_returns()
        for pod in self._pods:
            if not pod.is_active:
                continue
            pre_state = pod.state.value
            self._lifecycle.evaluate(
                pod, portfolio_returns_series, bar_timestamp=self._pending_bar_ts
            )

            # AssetSelector: all-excluded → WARNING 전이
            self._check_all_excluded_warning(pod)

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

            # GBM drawdown alert
            self._check_gbm_alert(pod, DrawdownSeverity)

    @staticmethod
    def _check_all_excluded_warning(pod: StrategyPod) -> None:
        """AssetSelector의 모든 에셋이 제외되면 WARNING으로 전이."""
        from src.orchestrator.models import LifecycleState

        selector = pod.asset_selector
        if selector is None:
            return
        if selector.all_excluded and pod.state == LifecycleState.PRODUCTION:
            pod.state = LifecycleState.WARNING
            logger.warning(
                "Pod [{}]: all assets excluded by AssetSelector → WARNING",
                pod.pod_id,
            )

    def _check_gbm_alert(self, pod: StrategyPod, drawdown_severity: type) -> None:
        """GBM drawdown 결과가 비정상이면 RiskAlertEvent 발행."""
        if self._lifecycle is None or self._bus is None:
            return

        gbm_result = self._lifecycle.get_gbm_result(pod.pod_id)
        if gbm_result is None or gbm_result.severity == drawdown_severity.NORMAL:
            return

        severity: str = (
            "CRITICAL" if gbm_result.severity == drawdown_severity.CRITICAL else "WARNING"
        )
        alert = RiskAlertEvent(
            alert_level=severity,  # type: ignore[arg-type]
            message=(
                f"GBM [{pod.pod_id}]: depth={gbm_result.current_depth:.1%} "
                f"(limit {gbm_result.expected_max_depth:.1%}), "
                f"duration={gbm_result.current_duration_days}d"
            ),
            source="GBMDrawdownMonitor",
        )
        self._fire_notification(self._bus.publish(alert))

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
            asset_price_history=self._close_price_history if self._close_price_history else None,
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
        if critical_alerts and not self._risk_breached:
            self._apply_risk_defense(critical_alerts)
        elif self._risk_breached:
            # 위기 해제 → 점진 복원 시작
            self._risk_breached = False
            self._risk_recovery_step = 1
            logger.info(
                "Risk defense deactivated — gradual recovery started (step 1/{})",
                self._config.risk_recovery_steps,
            )
        elif self._risk_recovery_step > 0:
            # 복원 진행 중
            self._risk_recovery_step += 1
            if self._risk_recovery_step > self._config.risk_recovery_steps:
                self._risk_recovery_step = 0
                logger.info("Risk recovery complete — full capital restored")
            else:
                logger.info(
                    "Risk recovery step {}/{}",
                    self._risk_recovery_step,
                    self._config.risk_recovery_steps,
                )

        return alerts

    def _apply_risk_defense(self, alerts: list[RiskAlert]) -> None:
        """CRITICAL alert 시 활성 Pod의 capital_fraction을 축소."""
        self._risk_breached = True
        self._risk_recovery_step = 0  # 복원 중 재위기 → 리셋
        for pod in self._pods:
            if pod.is_active:
                pod.capital_fraction *= _RISK_DEFENSE_SCALE
        logger.warning(
            "Risk defense: {} critical alerts, capital scaled by {:.0%}",
            len(alerts),
            _RISK_DEFENSE_SCALE,
        )

    # ── Daily Return MTM ─────────────────────────────────────────

    def set_initial_capital(self, capital: float) -> None:
        """초기 자본을 설정하고 각 Pod의 base equity를 초기화합니다.

        이미 복원된 base_equity가 있는 Pod은 건드리지 않습니다.

        Args:
            capital: 초기 자본 (USD)
        """
        self._initial_capital = capital
        for pod in self._pods:
            pod.set_base_equity(capital * pod.capital_fraction)

    def flush_daily_returns(self) -> None:
        """마지막 일 수익률을 강제 기록합니다 (백테스트/라이브 종료 시).

        _last_day_date가 None이면 (한 번도 기록되지 않았으면) 스킵합니다.
        """
        if self._last_day_date is None:
            return
        self._pending_day_change = False
        self._record_pod_daily_returns()

    def _record_pod_daily_returns(
        self,
        close_prices: dict[str, float] | None = None,
    ) -> None:
        """활성 Pod에 record_daily_return_mtm()을 호출합니다.

        Args:
            close_prices: 사용할 close price dict (None이면 _last_close_prices 사용)
        """
        prices = close_prices if close_prices is not None else self._last_close_prices
        for pod in self._pods:
            if not pod.is_active:
                continue
            pod_prices = {s: prices[s] for s in pod.symbols if s in prices}
            pod.record_daily_return_mtm(pod_prices)

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
    def initial_capital(self) -> float:
        """초기 자본 (USD)."""
        return self._initial_capital

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

    @property
    def last_pod_targets(self) -> dict[str, dict[str, float]]:
        """Pod별 마지막 타겟 가중치 (읽기 전용)."""
        return dict(self._last_pod_targets)

    @property
    def last_allocated_weights(self) -> dict[str, float]:
        """마지막 리밸런스에서 할당된 가중치 (읽기 전용)."""
        return dict(self._last_allocated_weights)

    # ── Serialization ──────────────────────────────────────────────

    def restore_histories(
        self,
        allocation_history: list[dict[str, object]],
        lifecycle_events: list[dict[str, object]],
        risk_contributions_history: list[dict[str, object]],
    ) -> None:
        """Restore in-memory histories from persistence layer."""
        self._allocation_history = allocation_history
        self._lifecycle_events = lifecycle_events
        self._risk_contributions_history = risk_contributions_history

    def to_dict(self) -> dict[str, object]:
        """Serialize orchestrator-level state for persistence.

        Histories are persisted separately via OrchestratorStatePersistence.
        """
        return {
            "last_rebalance_ts": (
                self._last_rebalance_ts.isoformat() if self._last_rebalance_ts is not None else None
            ),
            "last_pod_targets": {
                pod_id: dict(targets) for pod_id, targets in self._last_pod_targets.items()
            },
            "initial_capital": self._initial_capital,
            "last_day_date": self._last_day_date.isoformat()
            if self._last_day_date is not None
            else None,
            "last_close_prices": dict(self._last_close_prices),
            "risk_breached": self._risk_breached,
            "risk_recovery_step": self._risk_recovery_step,
            "close_price_history": {
                sym: list(prices) for sym, prices in self._close_price_history.items()
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

        # Daily return MTM state
        cap_val = data.get("initial_capital")
        if isinstance(cap_val, int | float):
            self._initial_capital = float(cap_val)

        day_val = data.get("last_day_date")
        if isinstance(day_val, str):
            self._last_day_date = date.fromisoformat(day_val)

        prices_val = data.get("last_close_prices")
        if isinstance(prices_val, dict):
            self._last_close_prices = {str(sym): float(p) for sym, p in prices_val.items()}

        risk_breached_val = data.get("risk_breached")
        if isinstance(risk_breached_val, bool):
            self._risk_breached = risk_breached_val

        recovery_val = data.get("risk_recovery_step")
        if isinstance(recovery_val, int | float):
            self._risk_recovery_step = int(recovery_val)

        price_hist_val = data.get("close_price_history")
        if isinstance(price_hist_val, dict):
            self._close_price_history = {
                str(sym): [float(p) for p in prices]
                for sym, prices in price_hist_val.items()
                if isinstance(prices, list)
            }

    # ── Leverage & Cleanup ────────────────────────────────────────

    def _apply_per_pod_leverage_cap(self) -> None:
        """Layer 2: Pod별 aggregate leverage를 max_leverage * capital_fraction으로 제한.

        _last_pod_targets의 weight를 비례 축소하여 Pod이 자기 한도를 초과하지 않도록 합니다.
        """
        for pod in self._pods:
            if not pod.is_active:
                continue
            pod_id = pod.pod_id
            pod_targets = self._last_pod_targets.get(pod_id)
            if not pod_targets:
                continue
            pod_gross = sum(abs(w) for w in pod_targets.values())
            pod_limit = pod.config.max_leverage * pod.capital_fraction
            if pod_gross > pod_limit and pod_gross > _MIN_NET_WEIGHT:
                scale = pod_limit / pod_gross
                self._last_pod_targets[pod_id] = {sym: w * scale for sym, w in pod_targets.items()}

    def _cleanup_retired_pod_targets(self) -> None:
        """E7: Retired pod의 잔여 타겟을 zero-out하여 phantom 포지션 방지."""
        from src.orchestrator.models import LifecycleState

        for pod in self._pods:
            if pod.state != LifecycleState.RETIRED:
                continue
            pod_targets = self._last_pod_targets.get(pod.pod_id)
            if pod_targets:
                self._last_pod_targets[pod.pod_id] = dict.fromkeys(pod_targets, 0.0)

    # ── Dynamic Universe ────────────────────────────────────────

    async def on_universe_update(
        self,
        scan_result: ScanResult,
        warmup_fn: Callable[..., Awaitable[tuple[list[dict[str, float]], list[datetime]]]],
    ) -> dict[str, list[str]]:
        """스캔 결과를 Pod에 적용.

        added → 모든 활성 Pod에 add_asset + warmup + routing
        dropped → AssetSelector permanently_excluded 플래그 + 버퍼 정리

        Args:
            scan_result: 스캔 결과
            warmup_fn: warmup bar fetcher (symbol, timeframe, limit) → (bars, timestamps)

        Returns:
            Pod별 추가된 심볼 {pod_id: [symbol, ...]}
        """
        pod_additions: dict[str, list[str]] = {}

        # 1. 신규 심볼 추가
        for symbol in scan_result.added:
            await self._add_symbol_to_pods(symbol, warmup_fn, pod_additions)

        # 2. 탈락 심볼 처리
        for symbol in scan_result.dropped:
            self._drop_symbol_from_pods(symbol)

        if pod_additions:
            logger.info("Universe update applied: {}", pod_additions)
        if scan_result.dropped:
            logger.info("Universe dropped symbols flagged: {}", scan_result.dropped)

        return pod_additions

    async def _add_symbol_to_pods(
        self,
        symbol: str,
        warmup_fn: Callable[..., Awaitable[tuple[list[dict[str, float]], list[datetime]]]],
        pod_additions: dict[str, list[str]],
    ) -> None:
        """신규 심볼을 모든 활성 Pod에 추가."""
        for idx, pod in enumerate(self._pods):
            if not pod.is_active:
                continue
            if not pod.add_asset(symbol):
                continue
            self._add_symbol_route(symbol, idx)
            try:
                bars, timestamps = await warmup_fn(symbol, pod.timeframe, pod.warmup_periods)
                if bars:
                    pod.inject_warmup(symbol, bars, timestamps)
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Pod [{}]: warmup failed for {}: {}",
                    pod.pod_id,
                    symbol,
                    exc,
                )
            pod_additions.setdefault(pod.pod_id, []).append(symbol)

    def _drop_symbol_from_pods(self, symbol: str) -> None:
        """탈락 심볼을 모든 활성 Pod에서 permanently_excluded 처리."""
        for pod in self._pods:
            if not pod.is_active or not pod.accepts_symbol(symbol):
                continue
            if pod.asset_selector is not None:
                pod.asset_selector.flag_permanently_excluded(symbol)
            pod.cleanup_excluded_asset(symbol)

    def _add_symbol_route(self, symbol: str, pod_idx: int) -> None:
        """심볼 라우팅 테이블에 단일 항목 추가.

        Args:
            symbol: 심볼
            pod_idx: Pod 인덱스
        """
        if symbol not in self._symbol_pod_map:
            self._symbol_pod_map[symbol] = []
        if pod_idx not in self._symbol_pod_map[symbol]:
            self._symbol_pod_map[symbol].append(pod_idx)

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
