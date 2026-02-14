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
    from src.orchestrator.allocator import CapitalAllocator
    from src.orchestrator.config import OrchestratorConfig
    from src.orchestrator.lifecycle import LifecycleManager
    from src.orchestrator.pod import StrategyPod
    from src.orchestrator.risk_aggregator import RiskAggregator

# ── Constants ─────────────────────────────────────────────────────

_ORCHESTRATOR_SOURCE = "StrategyOrchestrator"
_MIN_NET_WEIGHT = 1e-8


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
    ) -> None:
        self._config = config
        self._pods = pods
        self._allocator = allocator
        self._lifecycle = lifecycle_manager
        self._risk_aggregator = risk_aggregator
        self._bus: EventBus | None = None

        # 심볼 → Pod 인덱스 라우팅 테이블 (O(1) lookup)
        self._symbol_pod_map: dict[str, list[int]] = {}
        self._build_routing_table()

        # 배치 시그널
        self._pending_bar_ts: datetime | None = None
        self._pending_net_weights: dict[str, float] = {}

        # Fill attribution용: pod_id → {symbol → global_target}
        self._last_pod_targets: dict[str, dict[str, float]] = {}

        # 리밸런스 추적
        self._last_rebalance_ts: datetime | None = None

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
            symbol, fill.fill_qty, fill.fill_price, fill.fee, pod_targets,
        )

        for pod_id, (attr_qty, attr_price, attr_fee) in attributed.items():
            pod = self._find_pod(pod_id)
            if pod is not None:
                pod.update_position(
                    symbol, attr_qty, attr_price, attr_fee, is_buy=is_buy,
                )

    # ── Signal Emission ──────────────────────────────────────────

    async def _flush_net_signals(self) -> None:
        """누적된 net weight → SignalEvent 발행."""
        bus = self._bus
        if bus is None or self._pending_bar_ts is None:
            return

        # 레버리지 한도 적용
        scaled = scale_weights_to_leverage(
            self._pending_net_weights, self._config.max_gross_leverage,
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
        """현재 capital_fraction과 initial 비교하여 drift 확인."""
        for pod in self._pods:
            if not pod.is_active:
                continue
            drift = abs(pod.capital_fraction - pod.config.initial_fraction)
            if drift > self._config.rebalance_drift_threshold:
                return True
        return False

    def _execute_rebalance(self) -> None:
        """Allocator.compute_weights() 호출 → Pod capital_fraction 업데이트."""
        import pandas as pd

        # Lifecycle 평가 (weight 계산 전에 상태 전이)
        if self._lifecycle is not None:
            portfolio_returns_series = self._compute_portfolio_returns()
            for pod in self._pods:
                if pod.is_active:
                    self._lifecycle.evaluate(pod, portfolio_returns_series)

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

        new_weights = self._allocator.compute_weights(
            pod_returns,
            pod_states,
            lookback=self._config.correlation_lookback,
        )

        for pod in self._pods:
            if pod.pod_id in new_weights:
                pod.capital_fraction = new_weights[pod.pod_id]

        logger.debug("Rebalanced: {}", {p.pod_id: p.capital_fraction for p in self._pods})

        # Risk 한도 검사
        if self._risk_aggregator is not None:
            pod_performances = {
                pod.pod_id: pod.performance for pod in self._pods if pod.is_active
            }
            pod_weights_map = {
                pod.pod_id: pod.capital_fraction for pod in self._pods if pod.is_active
            }
            alerts = self._risk_aggregator.check_portfolio_limits(
                net_weights=self._pending_net_weights,
                pod_performances=pod_performances,
                pod_weights=pod_weights_map,
                pod_returns=pod_returns,
            )
            for alert in alerts:
                logger.warning("RiskAlert [{}]: {}", alert.severity, alert.message)

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
        """pod_id로 Pod 조회."""
        for pod in self._pods:
            if pod.pod_id == pod_id:
                return pod
        return None
