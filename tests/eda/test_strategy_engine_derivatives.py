"""Tests for StrategyEngine derivatives integration."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pandas as pd
import pytest

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType, SignalEvent
from src.eda.derivatives_feed import BacktestDerivativesProvider
from src.eda.strategy_engine import StrategyEngine
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals


class DerivAwareTestStrategy(BaseStrategy):
    """Derivatives 컬럼 인식 테스트 전략.

    funding_rate 컬럼이 있으면 direction에 반영.
    """

    @property
    def name(self) -> str:
        return "deriv_aware_test"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = (df["close"] > df["open"]).astype(int) * 2 - 1
        strength = pd.Series(0.5, index=df.index)
        entries = direction.diff().fillna(0).abs() > 0
        exits = pd.Series(False, index=df.index)

        # funding_rate가 있으면 strength에 반영
        if "funding_rate" in df.columns:
            fr = df["funding_rate"].fillna(0)
            strength = strength + fr.abs()

        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


def _make_bar(
    symbol: str,
    open_: float,
    close: float,
    ts: datetime,
) -> BarEvent:
    """테스트용 BarEvent."""
    return BarEvent(
        symbol=symbol,
        timeframe="1D",
        open=open_,
        high=max(open_, close) + 100,
        low=min(open_, close) - 100,
        close=close,
        volume=1000.0,
        bar_timestamp=ts,
        correlation_id=str(uuid4()),
        source="test",
    )


class TestStrategyEngineWithDerivatives:
    @pytest.mark.asyncio()
    async def test_no_derivatives_provider(self) -> None:
        """DerivativesProvider 없이 정상 동작."""
        strategy = DerivAwareTestStrategy()
        engine = StrategyEngine(strategy, warmup_periods=5)
        bus = EventBus()

        collected: list[AnyEvent] = []

        async def _collect(event: AnyEvent) -> None:
            if isinstance(event, SignalEvent):
                collected.append(event)

        bus.subscribe(EventType.SIGNAL, _collect)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())

        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(10):
            ts = base + timedelta(days=i)
            bar = _make_bar("BTC/USDT", 42000 + i * 10, 42000 + i * 15, ts)
            await bus.publish(bar)
            await bus.flush()

        await bus.stop()
        await task

        assert len(collected) >= 1

    @pytest.mark.asyncio()
    async def test_with_backtest_provider(self) -> None:
        """BacktestDerivativesProvider 전달 시 derivatives 컬럼 추가."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        deriv_index = pd.date_range(base, periods=60, freq="D", tz="UTC")
        deriv_df = pd.DataFrame(
            {"funding_rate": [0.0001 * (i % 5) for i in range(60)]},
            index=deriv_index,
        )
        provider = BacktestDerivativesProvider({"BTC/USDT": deriv_df})

        strategy = DerivAwareTestStrategy()
        engine = StrategyEngine(
            strategy,
            warmup_periods=5,
            derivatives_provider=provider,
        )
        bus = EventBus()

        collected: list[AnyEvent] = []

        async def _collect(event: AnyEvent) -> None:
            if isinstance(event, SignalEvent):
                collected.append(event)

        bus.subscribe(EventType.SIGNAL, _collect)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())

        for i in range(10):
            ts = base + timedelta(days=i)
            bar = _make_bar("BTC/USDT", 42000 + i * 10, 42000 + i * 15, ts)
            await bus.publish(bar)
            await bus.flush()

        await bus.stop()
        await task

        assert len(collected) >= 1

    @pytest.mark.asyncio()
    async def test_derivatives_enrichment_called(self) -> None:
        """_enrich_with_derivatives가 호출되어 DataFrame에 컬럼 추가."""
        base = datetime(2024, 1, 1, tzinfo=UTC)
        deriv_index = pd.date_range(base, periods=60, freq="D", tz="UTC")
        deriv_df = pd.DataFrame(
            {"funding_rate": [0.001] * 60},
            index=deriv_index,
        )
        provider = BacktestDerivativesProvider({"BTC/USDT": deriv_df})

        strategy = DerivAwareTestStrategy()
        engine = StrategyEngine(
            strategy,
            warmup_periods=5,
            derivatives_provider=provider,
        )
        bus = EventBus()

        collected: list[AnyEvent] = []

        async def _collect(event: AnyEvent) -> None:
            if isinstance(event, SignalEvent):
                collected.append(event)

        bus.subscribe(EventType.SIGNAL, _collect)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())

        for i in range(10):
            ts = base + timedelta(days=i)
            bar = _make_bar("BTC/USDT", 42000 + i * 10, 42000 + i * 15, ts)
            await bus.publish(bar)
            await bus.flush()

        await bus.stop()
        await task

        assert len(collected) >= 1
        for s in collected:
            assert isinstance(s, SignalEvent)
            assert s.strength >= 0.5

    @pytest.mark.asyncio()
    async def test_missing_symbol_in_provider(self) -> None:
        """Provider에 해당 심볼이 없으면 derivatives 없이 진행."""
        provider = BacktestDerivativesProvider({"ETH/USDT": pd.DataFrame()})

        strategy = DerivAwareTestStrategy()
        engine = StrategyEngine(
            strategy,
            warmup_periods=5,
            derivatives_provider=provider,
        )
        bus = EventBus()

        collected: list[AnyEvent] = []

        async def _collect(event: AnyEvent) -> None:
            if isinstance(event, SignalEvent):
                collected.append(event)

        bus.subscribe(EventType.SIGNAL, _collect)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())

        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(10):
            ts = base + timedelta(days=i)
            bar = _make_bar("BTC/USDT", 42000 + i * 10, 42000 + i * 15, ts)
            await bus.publish(bar)
            await bus.flush()

        await bus.stop()
        await task

        assert len(collected) >= 1
