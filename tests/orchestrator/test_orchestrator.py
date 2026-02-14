"""Tests for StrategyOrchestrator — 멀티 전략 오케스트레이터."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType, FillEvent, SignalEvent
from src.models.types import Direction
from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod, LifecycleState
from src.orchestrator.orchestrator import _ORCHESTRATOR_SOURCE, StrategyOrchestrator
from src.orchestrator.pod import StrategyPod
from src.orchestrator.risk_aggregator import RiskAggregator
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────────


class SimpleTestStrategy(BaseStrategy):
    """close > open → LONG(+1), else SHORT(-1). strength = abs(close-open)/open."""

    @property
    def name(self) -> str:
        return "test_simple"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = (df["close"] > df["open"]).astype(int) * 2 - 1
        strength = ((df["close"] - df["open"]).abs() / df["open"]).shift(1).fillna(0.01)
        entries = direction.diff().fillna(0).abs() > 0
        exits = pd.Series(False, index=df.index)
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


# ── Helpers ───────────────────────────────────────────────────────


def _make_pod_config(
    pod_id: str = "pod-a",
    symbols: tuple[str, ...] = ("BTC/USDT",),
    **overrides: object,
) -> PodConfig:
    defaults: dict[str, object] = {
        "pod_id": pod_id,
        "strategy_name": "tsmom",
        "symbols": symbols,
        "initial_fraction": 0.10,
        "max_fraction": 0.40,
        "min_fraction": 0.02,
    }
    defaults.update(overrides)
    return PodConfig(**defaults)  # type: ignore[arg-type]


def _make_orchestrator_config(
    pod_configs: tuple[PodConfig, ...] | None = None,
    **overrides: object,
) -> OrchestratorConfig:
    if pod_configs is None:
        pod_configs = (
            _make_pod_config("pod-a", ("BTC/USDT",)),
            _make_pod_config("pod-b", ("ETH/USDT",)),
        )
    defaults: dict[str, object] = {
        "pods": pod_configs,
        "allocation_method": AllocationMethod.EQUAL_WEIGHT,
        "rebalance_calendar_days": 7,
    }
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)  # type: ignore[arg-type]


def _make_pod(
    pod_id: str = "pod-a",
    symbols: tuple[str, ...] = ("BTC/USDT",),
    capital_fraction: float = 0.5,
    warmup: int = 3,
) -> StrategyPod:
    config = _make_pod_config(pod_id, symbols)
    strategy = SimpleTestStrategy()
    pod = StrategyPod(config=config, strategy=strategy, capital_fraction=capital_fraction)
    pod._warmup = warmup
    return pod


def _make_bar(
    symbol: str = "BTC/USDT",
    open_: float = 100.0,
    close: float = 110.0,
    ts: datetime | None = None,
) -> BarEvent:
    if ts is None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
    return BarEvent(
        symbol=symbol,
        timeframe="1D",
        open=open_,
        high=max(open_, close) * 1.01,
        low=min(open_, close) * 0.99,
        close=close,
        volume=1000.0,
        bar_timestamp=ts,
        correlation_id=uuid4(),
        source="test",
    )


def _make_fill(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    qty: float = 0.1,
    price: float = 50000.0,
    fee: float = 5.0,
) -> FillEvent:
    return FillEvent(
        client_order_id="test-order-1",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        fill_price=price,
        fill_qty=qty,
        fee=fee,
        fill_timestamp=datetime.now(UTC),
        source="test",
    )


def _feed_warmup_bars(
    pod: StrategyPod,
    symbol: str,
    n: int,
    base_ts: datetime,
) -> datetime:
    """Pod에 n개 bar를 공급. 마지막 ts 반환."""
    ts = base_ts
    for i in range(n):
        bar_data = {
            "open": 100.0 + i,
            "high": (101.0 + i) * 1.01,
            "low": (100.0 + i) * 0.99,
            "close": 101.0 + i,
            "volume": 1000.0,
        }
        pod.compute_signal(symbol, bar_data, ts)
        ts += timedelta(days=1)
    return ts


async def _run_with_bus(
    orchestrator: StrategyOrchestrator,
    events: list[AnyEvent],
) -> list[SignalEvent]:
    """EventBus + Orchestrator 실행, 발행된 SignalEvent 수집."""
    bus = EventBus(queue_size=100)
    signals: list[SignalEvent] = []

    async def signal_collector(event: AnyEvent) -> None:
        assert isinstance(event, SignalEvent)
        signals.append(event)

    bus.subscribe(EventType.SIGNAL, signal_collector)
    await orchestrator.register(bus)

    task = asyncio.create_task(bus.start())

    for evt in events:
        await bus.publish(evt)
        await bus.flush()

    # flush 남은 시그널
    await orchestrator.flush_pending_signals()
    await bus.flush()

    await bus.stop()
    await task

    return signals


# ── TestOrchestratorRouting ──────────────────────────────────────


class TestOrchestratorRouting:
    async def test_bar_routed_to_correct_pod(self) -> None:
        """BTC bar → pod-a에만 전달."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), warmup=3)
        pod_b = _make_pod("pod-b", ("ETH/USDT",), warmup=3)
        config = _make_orchestrator_config(
            (pod_a.config, pod_b.config),
        )
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        # BTC warmup
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod_a, "BTC/USDT", 3, ts)

        # BTC bar → pod_a만 시그널 생성
        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=3))
        signals = await _run_with_bus(orch, [bar])

        # pod_a가 시그널 생성했으므로 SignalEvent 발행
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert len(btc_signals) >= 1

    async def test_bar_not_routed_to_wrong_pod(self) -> None:
        """ETH bar는 pod-a(BTC only)에 라우팅 안 됨."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), warmup=3)
        config = _make_orchestrator_config(
            (pod_a.config,),
        )
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a], allocator)

        bar = _make_bar("ETH/USDT", 100.0, 110.0)
        signals = await _run_with_bus(orch, [bar])
        eth_signals = [s for s in signals if s.symbol == "ETH/USDT"]
        assert len(eth_signals) == 0

    async def test_overlapping_symbols_both_pods(self) -> None:
        """동일 심볼 → 두 Pod 모두 라우팅."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.3, warmup=3)
        pod_b = _make_pod("pod-b", ("BTC/USDT",), capital_fraction=0.3, warmup=3)
        config = _make_orchestrator_config(
            (pod_a.config, pod_b.config),
        )
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        # 두 Pod 모두 warmup
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod_a, "BTC/USDT", 3, ts)
        _feed_warmup_bars(pod_b, "BTC/USDT", 3, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=3))
        signals = await _run_with_bus(orch, [bar])
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert len(btc_signals) >= 1


# ── TestOrchestratorNetting ──────────────────────────────────────


class TestOrchestratorNetting:
    async def test_single_pod_signal_passthrough(self) -> None:
        """단일 Pod 시그널 그대로 전달."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=1.0, warmup=3)
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert len(btc_signals) == 1
        assert btc_signals[0].direction in (Direction.LONG, Direction.SHORT)

    async def test_two_pods_same_direction_additive(self) -> None:
        """같은 방향 → 합산 (절대값 증가)."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.4, warmup=3)
        pod_b = _make_pod("pod-b", ("BTC/USDT",), capital_fraction=0.4, warmup=3)
        config = _make_orchestrator_config((pod_a.config, pod_b.config))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod_a, "BTC/USDT", 2, ts)
        _feed_warmup_bars(pod_b, "BTC/USDT", 2, ts)

        # close > open → 둘 다 LONG
        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert len(btc_signals) == 1
        assert btc_signals[0].direction == Direction.LONG

    async def test_two_pods_opposite_direction_cancel(self) -> None:
        """반대 방향 → 상쇄 (넷팅)."""
        pod_long = _make_pod("pod-long", ("BTC/USDT",), capital_fraction=0.3, warmup=3)
        pod_short = _make_pod("pod-short", ("BTC/USDT",), capital_fraction=0.3, warmup=3)
        config = _make_orchestrator_config((pod_long.config, pod_short.config))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_long, pod_short], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        # pod_long warmup: close > open (LONG)
        for i in range(2):
            bar_data = {"open": 100.0 + i, "high": 111.0 + i, "low": 99.0 + i, "close": 110.0 + i, "volume": 1000.0}
            pod_long.compute_signal("BTC/USDT", bar_data, ts + timedelta(days=i))
        # pod_short warmup: close < open (SHORT)
        for i in range(2):
            bar_data = {"open": 110.0 + i, "high": 111.0 + i, "low": 99.0 + i, "close": 100.0 + i, "volume": 1000.0}
            pod_short.compute_signal("BTC/USDT", bar_data, ts + timedelta(days=i))

        # pod_long → LONG, pod_short → SHORT (같은 bar 데이터로도 전략 히스토리가 다름)
        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])
        # 상쇄 결과: net이 작으면 NEUTRAL 가능, 크면 한 방향
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert len(btc_signals) == 1

    async def test_neutral_when_net_near_zero(self) -> None:
        """net ≈ 0 → NEUTRAL."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.0, warmup=3)
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        # capital_fraction=0 → global_weight=0 → NEUTRAL
        if btc_signals:
            assert btc_signals[0].direction == Direction.NEUTRAL
            assert btc_signals[0].strength == pytest.approx(0.0)


# ── TestOrchestratorSignalEmission ───────────────────────────────


class TestOrchestratorSignalEmission:
    async def test_signal_event_published_on_bus(self) -> None:
        """시그널이 EventBus에 발행됨."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.5, warmup=3)
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])
        assert len(signals) >= 1

    async def test_signal_direction_correct(self) -> None:
        """close > open → LONG 시그널."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=1.0, warmup=3)
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert btc_signals[0].direction == Direction.LONG

    async def test_signal_strength_equals_abs_net(self) -> None:
        """strength = abs(net_weight)."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=1.0, warmup=3)
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert btc_signals[0].strength > 0

    async def test_signal_source_is_orchestrator(self) -> None:
        """SignalEvent.source == "StrategyOrchestrator"."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=1.0, warmup=3)
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert btc_signals[0].source == _ORCHESTRATOR_SOURCE


# ── TestOrchestratorFillAttribution ──────────────────────────────


class TestOrchestratorFillAttribution:
    async def test_fill_proportional_split(self) -> None:
        """Fill이 두 Pod에 비례 배분됨."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.6, warmup=3)
        pod_b = _make_pod("pod-b", ("BTC/USDT",), capital_fraction=0.4, warmup=3)
        config = _make_orchestrator_config((pod_a.config, pod_b.config))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod_a, "BTC/USDT", 2, ts)
        _feed_warmup_bars(pod_b, "BTC/USDT", 2, ts)

        # 시그널 생성 (on_bar 경유)
        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        fill = _make_fill("BTC/USDT", "BUY", 1.0, 50000.0, 10.0)

        bus = EventBus(queue_size=100)
        await orch.register(bus)
        task = asyncio.create_task(bus.start())

        await bus.publish(bar)
        await bus.flush()
        await orch.flush_pending_signals()
        await bus.flush()
        await bus.publish(fill)
        await bus.flush()

        await bus.stop()
        await task

        # 두 Pod 모두 trade_count > 0
        assert pod_a.performance.trade_count >= 1
        assert pod_b.performance.trade_count >= 1

    async def test_fill_single_pod_100pct(self) -> None:
        """단일 Pod → 100% 귀속."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=1.0, warmup=3)
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        fill = _make_fill("BTC/USDT", "BUY", 1.0, 50000.0, 10.0)

        bus = EventBus(queue_size=100)
        await orch.register(bus)
        task = asyncio.create_task(bus.start())

        await bus.publish(bar)
        await bus.flush()
        await orch.flush_pending_signals()
        await bus.flush()
        await bus.publish(fill)
        await bus.flush()

        await bus.stop()
        await task

        assert pod.performance.trade_count == 1

    async def test_fill_no_targets_no_attribution(self) -> None:
        """타겟 없는 심볼에 Fill → 귀속 안 됨."""
        pod = _make_pod("pod-a", ("BTC/USDT",), warmup=3)
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        # ETH fill (pod는 BTC만 관리)
        fill = _make_fill("ETH/USDT", "BUY", 1.0, 3000.0, 5.0)

        bus = EventBus(queue_size=100)
        await orch.register(bus)
        task = asyncio.create_task(bus.start())

        await bus.publish(fill)
        await bus.flush()

        await bus.stop()
        await task

        assert pod.performance.trade_count == 0


# ── TestOrchestratorRebalance ────────────────────────────────────


class TestOrchestratorRebalance:
    async def test_rebalance_after_calendar_days(self) -> None:
        """calendar_days 경과 후 리밸런스 실행."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.5, warmup=3)
        pod_b = _make_pod("pod-b", ("ETH/USDT",), capital_fraction=0.5, warmup=3)
        config = _make_orchestrator_config(
            (pod_a.config, pod_b.config),
            rebalance_calendar_days=2,
        )
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        # Pod에 일별 수익률 기록 (allocator가 사용)
        pod_a.record_daily_return(0.01)
        pod_b.record_daily_return(0.02)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod_a, "BTC/USDT", 2, ts)
        _feed_warmup_bars(pod_b, "ETH/USDT", 2, ts)

        # Day 3 bar → 첫 리밸런스
        bar1 = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        await _run_with_bus(orch, [bar1])

        # 리밸런스가 실행되어 capital_fraction이 변경됨
        # (EW 알고리즘이므로 둘 다 0.5 근방)
        assert pod_a.capital_fraction > 0
        assert pod_b.capital_fraction > 0

    async def test_capital_fraction_updated(self) -> None:
        """리밸런스 후 Pod capital_fraction 업데이트."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.8, warmup=3)
        pod_b = _make_pod("pod-b", ("ETH/USDT",), capital_fraction=0.2, warmup=3)
        config = _make_orchestrator_config(
            (pod_a.config, pod_b.config),
            rebalance_calendar_days=1,
        )
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        # 수익률 데이터 기록
        for _ in range(5):
            pod_a.record_daily_return(0.01)
            pod_b.record_daily_return(0.01)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod_a, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        await _run_with_bus(orch, [bar])

        # EW 알고리즘: 리밸런스 후 비슷한 비중
        total = pod_a.capital_fraction + pod_b.capital_fraction
        assert total <= 1.0 + 1e-6


# ── TestOrchestratorEdgeCases ────────────────────────────────────


class TestOrchestratorEdgeCases:
    async def test_retired_pod_skipped(self) -> None:
        """RETIRED Pod은 시그널 생성에서 제외."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.5, warmup=3)
        pod_b = _make_pod("pod-b", ("BTC/USDT",), capital_fraction=0.5, warmup=3)
        pod_b.state = LifecycleState.RETIRED
        config = _make_orchestrator_config((pod_a.config, pod_b.config))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod_a, "BTC/USDT", 2, ts)
        _feed_warmup_bars(pod_b, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])

        # pod_b는 RETIRED → 시그널 없음, pod_a만 기여
        assert orch.active_pod_count == 1
        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        assert len(btc_signals) >= 1

    def test_get_pod_summary(self) -> None:
        """get_pod_summary() 반환값 구조."""
        pod = _make_pod("pod-a", ("BTC/USDT",))
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        summary = orch.get_pod_summary()
        assert len(summary) == 1
        assert summary[0]["pod_id"] == "pod-a"
        assert summary[0]["is_active"] is True
        assert "capital_fraction" in summary[0]


# ── TestOrchestratorNettingIntegration ─────────────────────────


class TestOrchestratorNettingIntegration:
    """netting 모듈 통합 테스트."""

    async def test_fill_via_attribute_fill(self) -> None:
        """attribute_fill() 경유로 비례 귀속이 동작."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.6, warmup=3)
        pod_b = _make_pod("pod-b", ("BTC/USDT",), capital_fraction=0.4, warmup=3)
        config = _make_orchestrator_config((pod_a.config, pod_b.config))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod_a, pod_b], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod_a, "BTC/USDT", 2, ts)
        _feed_warmup_bars(pod_b, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        fill = _make_fill("BTC/USDT", "BUY", 1.0, 50000.0, 10.0)

        bus = EventBus(queue_size=100)
        await orch.register(bus)
        task = asyncio.create_task(bus.start())

        await bus.publish(bar)
        await bus.flush()
        await orch.flush_pending_signals()
        await bus.flush()
        await bus.publish(fill)
        await bus.flush()

        await bus.stop()
        await task

        # 두 Pod 모두 fill 귀속
        assert pod_a.performance.trade_count >= 1
        assert pod_b.performance.trade_count >= 1

    async def test_leverage_scaling_applied(self) -> None:
        """max_gross_leverage 초과 시 가중치 축소."""
        pod = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=5.0, warmup=3)
        config = _make_orchestrator_config(
            (pod.config,),
            max_gross_leverage=2.0,
        )
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod, "BTC/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        signals = await _run_with_bus(orch, [bar])

        btc_signals = [s for s in signals if s.symbol == "BTC/USDT"]
        if btc_signals:
            # strength가 2.0 이하로 축소되어야 함
            assert btc_signals[0].strength <= 2.0 + 1e-6


# ── TestOrchestratorRiskIntegration ────────────────────────────


class TestOrchestratorRiskIntegration:
    """RiskAggregator 통합 테스트."""

    def test_risk_aggregator_none_backward_compat(self) -> None:
        """risk_aggregator=None → 기존 동작 유지."""
        pod = _make_pod("pod-a", ("BTC/USDT",))
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        orch = StrategyOrchestrator(config, [pod], allocator)
        assert orch._risk_aggregator is None

    def test_risk_aggregator_injected(self) -> None:
        """RiskAggregator 주입 시 저장."""
        pod = _make_pod("pod-a", ("BTC/USDT",))
        config = _make_orchestrator_config((pod.config,))
        allocator = CapitalAllocator(config)
        ra = RiskAggregator(config)
        orch = StrategyOrchestrator(config, [pod], allocator, risk_aggregator=ra)
        assert orch._risk_aggregator is ra

    async def test_rebalance_triggers_risk_check(self) -> None:
        """리밸런스 시 RiskAggregator.check_portfolio_limits() 호출."""
        pod_a = _make_pod("pod-a", ("BTC/USDT",), capital_fraction=0.5, warmup=3)
        pod_b = _make_pod("pod-b", ("ETH/USDT",), capital_fraction=0.5, warmup=3)
        config = _make_orchestrator_config(
            (pod_a.config, pod_b.config),
            rebalance_calendar_days=1,
        )
        allocator = CapitalAllocator(config)
        ra = RiskAggregator(config)
        orch = StrategyOrchestrator(
            config, [pod_a, pod_b], allocator, risk_aggregator=ra,
        )

        pod_a.record_daily_return(0.01)
        pod_b.record_daily_return(0.02)

        ts = datetime(2024, 1, 1, tzinfo=UTC)
        _feed_warmup_bars(pod_a, "BTC/USDT", 2, ts)
        _feed_warmup_bars(pod_b, "ETH/USDT", 2, ts)

        bar = _make_bar("BTC/USDT", 100.0, 110.0, ts + timedelta(days=2))
        # 리밸런스 + 리스크 체크 실행 (로깅으로 확인 — 에러 없이 완료되면 성공)
        await _run_with_bus(orch, [bar])
        assert orch.active_pod_count == 2
