"""StrategyEngine + RegimeService 통합 테스트.

RegimeService가 StrategyEngine에 주입될 때의 동작을 검증합니다.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pandas as pd

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType
from src.eda.strategy_engine import StrategyEngine
from src.regime.service import REGIME_COLUMNS, RegimeService, RegimeServiceConfig
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ──


class RegimeAwareStrategy(BaseStrategy):
    """regime 컬럼을 사용하는 테스트 전략.

    regime_label이 있으면 사용, 없으면 기본 동작.
    """

    seen_regime_columns: list[set[str]] = []

    @property
    def name(self) -> str:
        return "test_regime_aware"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        # regime 컬럼 존재 여부 기록
        regime_cols = {c for c in df.columns if c in set(REGIME_COLUMNS)}
        RegimeAwareStrategy.seen_regime_columns.append(regime_cols)

        direction = pd.Series(1, index=df.index)
        strength = pd.Series(1.0, index=df.index)
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)


class SimpleStrategy(BaseStrategy):
    """regime과 무관한 간단한 전략."""

    @property
    def name(self) -> str:
        return "test_simple"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = pd.Series(1, index=df.index)
        strength = pd.Series(1.0, index=df.index)
        entries = pd.Series(True, index=df.index)
        exits = pd.Series(False, index=df.index)
        return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)


def _make_bar(
    symbol: str,
    close: float,
    ts: datetime,
    timeframe: str = "1D",
) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timeframe=timeframe,
        open=close * 0.99,
        high=close * 1.01,
        low=close * 0.98,
        close=close,
        volume=1000.0,
        bar_timestamp=ts,
        correlation_id=uuid4(),
        source="test",
    )


def _warmup_bars(n: int, start_close: float = 100.0) -> list[BarEvent]:
    """warmup용 bar 시리즈 생성."""
    bars = []
    rng = np.random.default_rng(42)
    close = start_close
    base = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(n):
        close *= 1 + rng.normal(0.005, 0.01)
        bars.append(_make_bar("BTC/USDT", close, base + timedelta(days=i)))
    return bars


class TestRegimeServiceNone:
    """regime_service=None 시 기존 동작 유지."""

    async def test_no_regime_columns_without_service(self) -> None:
        """regime_service=None이면 regime 컬럼 미추가."""
        RegimeAwareStrategy.seen_regime_columns = []

        strategy = RegimeAwareStrategy()
        engine = StrategyEngine(strategy, warmup_periods=3, regime_service=None)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        signals: list[AnyEvent] = []
        bus.subscribe(EventType.SIGNAL, lambda e: signals.append(e))
        bus_task = asyncio.create_task(bus.start())

        bars = _warmup_bars(5)
        for bar in bars:
            await bus.publish(bar)
        await bus.flush()

        await bus.stop()
        await bus_task

        # SignalEvent이 발행됨
        assert len(signals) > 0
        # regime 컬럼 없음
        for cols in RegimeAwareStrategy.seen_regime_columns:
            assert len(cols) == 0


class TestRegimeServiceIntegration:
    """regime_service가 주입된 상태에서 StrategyEngine 동작."""

    async def test_precomputed_regime_enriches_df(self) -> None:
        """사전 계산된 regime이 df에 추가됨."""
        RegimeAwareStrategy.seen_regime_columns = []

        # RegimeService + precompute
        service = RegimeServiceConfig(target_timeframe="1D")
        regime_service = RegimeService(service)

        bars = _warmup_bars(30)
        closes = pd.Series(
            [bar.close for bar in bars],
            index=pd.DatetimeIndex([bar.bar_timestamp for bar in bars]),
        )
        regime_service.precompute("BTC/USDT", closes)

        strategy = RegimeAwareStrategy()
        engine = StrategyEngine(
            strategy,
            warmup_periods=3,
            regime_service=regime_service,
        )
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        signals: list[AnyEvent] = []
        bus.subscribe(EventType.SIGNAL, lambda e: signals.append(e))
        bus_task = asyncio.create_task(bus.start())

        for bar in bars:
            await bus.publish(bar)
        await bus.flush()

        await bus.stop()
        await bus_task

        assert len(signals) > 0
        # regime 컬럼이 추가되었어야 함
        has_regime = any(len(cols) > 0 for cols in RegimeAwareStrategy.seen_regime_columns)
        assert has_regime

    async def test_live_fallback_broadcast(self) -> None:
        """사전 계산 없이 live fallback으로 regime 컬럼 broadcast."""
        RegimeAwareStrategy.seen_regime_columns = []

        # RegimeService + warmup (precompute 없이)
        config = RegimeServiceConfig(target_timeframe="1D")
        regime_service = RegimeService(config)

        bars = _warmup_bars(50)
        # warmup으로 detector 초기화 (30개 bar)
        warmup_closes = [bar.close for bar in bars[:30]]
        regime_service.warmup("BTC/USDT", warmup_closes)

        strategy = RegimeAwareStrategy()
        engine = StrategyEngine(
            strategy,
            warmup_periods=3,
            regime_service=regime_service,
        )
        bus = EventBus(queue_size=100)
        await regime_service.register(bus)
        await engine.register(bus)

        signals: list[AnyEvent] = []
        bus.subscribe(EventType.SIGNAL, lambda e: signals.append(e))
        bus_task = asyncio.create_task(bus.start())

        # warmup 이후 bar 전송
        for bar in bars[30:]:
            await bus.publish(bar)
            await bus.flush()

        await bus.stop()
        await bus_task

        assert len(signals) > 0
        # live fallback으로 regime 컬럼이 추가되었어야 함
        has_regime = any(len(cols) > 0 for cols in RegimeAwareStrategy.seen_regime_columns)
        assert has_regime

    async def test_non_regime_strategy_unaffected(self) -> None:
        """regime 무관 전략에 regime 컬럼이 추가되어도 정상 동작."""
        config = RegimeServiceConfig(target_timeframe="1D")
        regime_service = RegimeService(config)

        bars = _warmup_bars(30)
        closes = pd.Series(
            [bar.close for bar in bars],
            index=pd.DatetimeIndex([bar.bar_timestamp for bar in bars]),
        )
        regime_service.precompute("BTC/USDT", closes)

        strategy = SimpleStrategy()
        engine = StrategyEngine(
            strategy,
            warmup_periods=3,
            regime_service=regime_service,
        )
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        signals: list[AnyEvent] = []
        bus.subscribe(EventType.SIGNAL, lambda e: signals.append(e))
        bus_task = asyncio.create_task(bus.start())

        for bar in bars:
            await bus.publish(bar)
        await bus.flush()

        await bus.stop()
        await bus_task

        # 정상적으로 SignalEvent 발행
        assert len(signals) > 0

    async def test_regime_registration_order(self) -> None:
        """RegimeService가 StrategyEngine 앞에 등록되어야 BAR 처리 순서 보장."""
        config = RegimeServiceConfig(target_timeframe="1D")
        regime_service = RegimeService(config)
        strategy = SimpleStrategy()
        engine = StrategyEngine(strategy, warmup_periods=3, regime_service=regime_service)

        bus = EventBus(queue_size=100)

        # regime → strategy 순서로 등록
        await regime_service.register(bus)
        await engine.register(bus)

        handlers = bus._handlers[EventType.BAR]
        # RegimeService._on_bar가 StrategyEngine._on_bar 앞에 등록됨
        assert len(handlers) >= 2
        # 튜플 구조: (priority, seq, handler)
        assert handlers[0][2] == regime_service._on_bar
        assert handlers[1][2] == engine._on_bar
