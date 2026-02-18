"""Tests for StrategyOrchestrator Multi-TF routing.

Pod-A(4h) + Pod-B(1D) → 4h bar: A만 signal, 1D bar: B만 signal.
같은 심볼 공유 시 net weight = A + B.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType, SignalEvent
from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod
from src.orchestrator.orchestrator import StrategyOrchestrator
from src.orchestrator.pod import StrategyPod
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────────


class AlwaysLongStrategy(BaseStrategy):
    """항상 LONG(+1), strength=0.5."""

    @property
    def name(self) -> str:
        return "always_long"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = pd.Series(1, index=df.index)
        strength = pd.Series(0.5, index=df.index)
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


# ── Helpers ───────────────────────────────────────────────────────


def _make_pod_config(
    pod_id: str,
    symbols: tuple[str, ...],
    timeframe: str = "1D",
    initial_fraction: float = 0.5,
) -> PodConfig:
    return PodConfig(
        pod_id=pod_id,
        strategy_name="always_long",
        symbols=symbols,
        timeframe=timeframe,
        initial_fraction=initial_fraction,
        max_fraction=0.70,
        min_fraction=0.02,
    )


def _make_orch_config(pods: tuple[PodConfig, ...]) -> OrchestratorConfig:
    return OrchestratorConfig(
        pods=pods,
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
        rebalance_calendar_days=30,
    )


def _make_pod(config: PodConfig, warmup: int = 2) -> StrategyPod:
    strategy = AlwaysLongStrategy()
    pod = StrategyPod(config=config, strategy=strategy, capital_fraction=config.initial_fraction)
    pod._warmup = warmup
    return pod


def _make_bar(
    symbol: str,
    timeframe: str = "1D",
    ts: datetime | None = None,
) -> BarEvent:
    if ts is None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
    return BarEvent(
        symbol=symbol,
        timeframe=timeframe,
        open=100.0,
        high=110.0,
        low=99.0,
        close=105.0,
        volume=1000.0,
        bar_timestamp=ts,
        correlation_id=uuid4(),
        source="test",
    )


def _feed_warmup(pod: StrategyPod, symbol: str, n: int, base_ts: datetime) -> datetime:
    ts = base_ts
    for i in range(n):
        bar_data = {
            "open": 100.0 + i,
            "high": (100.0 + i) * 1.01,
            "low": (100.0 + i) * 0.99,
            "close": 101.0 + i,
            "volume": 1000.0,
        }
        pod.compute_signal(symbol, bar_data, ts)
        ts += timedelta(hours=4)
    return ts


async def _collect_signals(
    orchestrator: StrategyOrchestrator,
    events: list[BarEvent],
) -> list[SignalEvent]:
    """EventBus + Orchestrator 실행, 발행된 SignalEvent 수집."""
    bus = EventBus(queue_size=100)
    signals: list[SignalEvent] = []

    async def signal_collector(event: AnyEvent) -> None:
        assert isinstance(event, SignalEvent)
        signals.append(event)

    bus.subscribe(EventType.SIGNAL, signal_collector)
    await orchestrator.register(bus)
    bus_task = asyncio.create_task(bus.start())

    for event in events:
        await bus.publish(event)
        await bus.flush()

    # Flush pending signals
    await orchestrator.flush_pending_signals()
    await bus.flush()

    await bus.stop()
    await bus_task
    return signals


# ── Tests ────────────────────────────────────────────────────────


class TestMultiTFRouting:
    """Multi-TF per-pod TF 라우팅 테스트."""

    @pytest.mark.asyncio
    async def test_pod_only_receives_matching_tf_bars(self) -> None:
        """4h Pod은 4h bar만, 1D Pod은 1D bar만 수신."""
        pod_a_cfg = _make_pod_config("pod-a", ("BTC/USDT",), timeframe="4h")
        pod_b_cfg = _make_pod_config("pod-b", ("BTC/USDT",), timeframe="1D")
        orch_config = _make_orch_config((pod_a_cfg, pod_b_cfg))

        pod_a = _make_pod(pod_a_cfg, warmup=2)
        pod_b = _make_pod(pod_b_cfg, warmup=2)

        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[pod_a, pod_b],
            allocator=allocator,
            target_timeframe=None,  # Multi-TF mode
        )

        base_ts = datetime(2024, 1, 1, tzinfo=UTC)
        # Warmup: 4h pod
        _feed_warmup(pod_a, "BTC/USDT", 3, base_ts)
        # Warmup: 1D pod
        _feed_warmup(pod_b, "BTC/USDT", 3, base_ts)

        # 4h bar → Pod-A만 signal 생성
        ts1 = base_ts + timedelta(hours=12)
        signals = await _collect_signals(
            orchestrator,
            [
                _make_bar("BTC/USDT", "4h", ts1),
            ],
        )

        # Pod-A가 signal 생성 (warmup 이후)
        assert len(signals) >= 1
        # 모든 signal의 source는 Orchestrator
        assert all(s.strategy_name == "StrategyOrchestrator" for s in signals)

    @pytest.mark.asyncio
    async def test_accepted_timeframes_set(self) -> None:
        """target_timeframe=None → accepted_timeframes = Pod TF 집합."""
        pod_a_cfg = _make_pod_config("pod-a", ("BTC/USDT",), timeframe="4h")
        pod_b_cfg = _make_pod_config("pod-b", ("ETH/USDT",), timeframe="1D")
        orch_config = _make_orch_config((pod_a_cfg, pod_b_cfg))

        pod_a = _make_pod(pod_a_cfg)
        pod_b = _make_pod(pod_b_cfg)

        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[pod_a, pod_b],
            allocator=allocator,
            target_timeframe=None,
        )

        assert orchestrator._accepted_timeframes == {"4h", "1D"}

    @pytest.mark.asyncio
    async def test_single_tf_backward_compat(self) -> None:
        """target_timeframe='1D' → accepted_timeframes = {'1D'} (기존 호환)."""
        pod_cfg = _make_pod_config("pod-a", ("BTC/USDT",), timeframe="1D")
        orch_config = _make_orch_config((pod_cfg,))

        pod = _make_pod(pod_cfg)
        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[pod],
            allocator=allocator,
            target_timeframe="1D",
        )

        assert orchestrator._accepted_timeframes == {"1D"}

    @pytest.mark.asyncio
    async def test_1m_bar_ignored(self) -> None:
        """1m bar는 accepted_timeframes에 없으므로 무시."""
        pod_cfg = _make_pod_config("pod-a", ("BTC/USDT",), timeframe="1D")
        orch_config = _make_orch_config((pod_cfg,))

        pod = _make_pod(pod_cfg)
        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[pod],
            allocator=allocator,
            target_timeframe=None,
        )

        # 1m bar → no signal
        base_ts = datetime(2024, 1, 1, tzinfo=UTC)
        signals = await _collect_signals(
            orchestrator,
            [
                _make_bar("BTC/USDT", "1m", base_ts),
            ],
        )
        assert signals == []


class TestMultiTFNetWeight:
    """Multi-TF net weight 계산 테스트."""

    @pytest.mark.asyncio
    async def test_shared_symbol_net_weight(self) -> None:
        """같은 심볼을 4h + 1D Pod이 공유 → net weight = 합산."""
        pod_a_cfg = _make_pod_config("pod-a", ("BTC/USDT",), timeframe="4h", initial_fraction=0.3)
        pod_b_cfg = _make_pod_config("pod-b", ("BTC/USDT",), timeframe="1D", initial_fraction=0.3)
        orch_config = _make_orch_config((pod_a_cfg, pod_b_cfg))

        pod_a = _make_pod(pod_a_cfg, warmup=2)
        pod_b = _make_pod(pod_b_cfg, warmup=2)

        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[pod_a, pod_b],
            allocator=allocator,
            target_timeframe=None,
        )

        base_ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup(pod_a, "BTC/USDT", 3, base_ts)
        _feed_warmup(pod_b, "BTC/USDT", 3, base_ts)

        # 4h bar → Pod-A signal → _last_pod_targets 저장
        ts1 = base_ts + timedelta(hours=12)
        ts2 = base_ts + timedelta(days=1)

        bus = EventBus(queue_size=100)
        signals: list[SignalEvent] = []

        async def collector(event: AnyEvent) -> None:
            assert isinstance(event, SignalEvent)
            signals.append(event)

        bus.subscribe(EventType.SIGNAL, collector)
        await orchestrator.register(bus)
        bus_task = asyncio.create_task(bus.start())

        # 4h bar
        await bus.publish(_make_bar("BTC/USDT", "4h", ts1))
        await bus.flush()
        await orchestrator.flush_pending_signals()
        await bus.flush()

        # 1D bar (different timestamp → flush 4h pending + process 1D)
        await bus.publish(_make_bar("BTC/USDT", "1D", ts2))
        await bus.flush()
        await orchestrator.flush_pending_signals()
        await bus.flush()

        await bus.stop()
        await bus_task

        # 두 번째 signal은 4h + 1D 합산 net weight
        # _last_pod_targets에 양쪽 Pod의 기여분이 있어야 함
        assert len(orchestrator._last_pod_targets) == 2
        assert "pod-a" in orchestrator._last_pod_targets
        assert "pod-b" in orchestrator._last_pod_targets


class TestOrchestratorConfigAllTimeframes:
    """OrchestratorConfig.all_timeframes 프로퍼티 테스트."""

    def test_all_same_tf(self) -> None:
        """모든 Pod이 동일 TF → 1개만 반환."""
        config = _make_orch_config(
            (
                _make_pod_config("pod-a", ("BTC/USDT",), timeframe="1D"),
                _make_pod_config("pod-b", ("ETH/USDT",), timeframe="1D"),
            )
        )
        assert config.all_timeframes == ("1D",)

    def test_different_tfs(self) -> None:
        """Pod별 다른 TF → 순서 보존."""
        config = _make_orch_config(
            (
                _make_pod_config("pod-a", ("BTC/USDT",), timeframe="1D"),
                _make_pod_config("pod-b", ("ETH/USDT",), timeframe="4h"),
            )
        )
        assert config.all_timeframes == ("1D", "4h")

    def test_three_pods_two_tfs(self) -> None:
        """3 Pod, 2 TF → 중복 제거."""
        config = OrchestratorConfig(
            pods=(
                PodConfig(
                    pod_id="pod-a",
                    strategy_name="t",
                    symbols=("BTC/USDT",),
                    timeframe="1D",
                    initial_fraction=0.3,
                    max_fraction=0.40,
                    min_fraction=0.02,
                ),
                PodConfig(
                    pod_id="pod-b",
                    strategy_name="t",
                    symbols=("ETH/USDT",),
                    timeframe="4h",
                    initial_fraction=0.3,
                    max_fraction=0.40,
                    min_fraction=0.02,
                ),
                PodConfig(
                    pod_id="pod-c",
                    strategy_name="t",
                    symbols=("SOL/USDT",),
                    timeframe="1D",
                    initial_fraction=0.3,
                    max_fraction=0.40,
                    min_fraction=0.02,
                ),
            ),
            allocation_method=AllocationMethod.EQUAL_WEIGHT,
            rebalance_calendar_days=30,
        )
        assert config.all_timeframes == ("1D", "4h")


class TestPodTimeframeProperty:
    """StrategyPod.timeframe 프로퍼티 테스트."""

    def test_timeframe_matches_config(self) -> None:
        pod_cfg = _make_pod_config("pod-a", ("BTC/USDT",), timeframe="12H")
        pod = _make_pod(pod_cfg)
        assert pod.timeframe == "12H"

    def test_default_timeframe(self) -> None:
        pod_cfg = _make_pod_config("pod-a", ("BTC/USDT",))
        pod = _make_pod(pod_cfg)
        assert pod.timeframe == "1D"
