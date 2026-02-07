"""HistoricalDataFeed 테스트.

단일/멀티 심볼 데이터의 BarEvent 리플레이를 검증합니다.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType
from src.data.market_data import MarketDataSet, MultiSymbolData
from src.eda.data_feed import HistoricalDataFeed


def _make_ohlcv(n: int = 10, base_price: float = 50000.0) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame 생성."""
    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = pd.date_range(start=start, periods=n, freq="1D", tz=UTC)

    close = base_price + np.cumsum(rng.standard_normal(n) * 100)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100, 1000, n) * 1000.0,
        },
        index=timestamps,
    )


def _make_single_data(n: int = 10) -> MarketDataSet:
    """단일 심볼 MarketDataSet 생성."""
    df = _make_ohlcv(n)
    return MarketDataSet(
        symbol="BTC/USDT",
        timeframe="1D",
        start=df.index[0].to_pydatetime(),  # type: ignore[union-attr]
        end=df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
        ohlcv=df,
    )


def _make_multi_data(symbols: list[str], n: int = 10) -> MultiSymbolData:
    """멀티 심볼 MultiSymbolData 생성."""
    ohlcv: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        ohlcv[sym] = _make_ohlcv(n, base_price=50000.0 + i * 10000)

    first_df = ohlcv[symbols[0]]
    return MultiSymbolData(
        symbols=symbols,
        timeframe="1D",
        start=first_df.index[0].to_pydatetime(),  # type: ignore[union-attr]
        end=first_df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
        ohlcv=ohlcv,
    )


class TestHistoricalDataFeedSingle:
    """단일 심볼 리플레이 테스트."""

    async def test_correct_bar_count(self) -> None:
        """발행된 BarEvent 수가 DataFrame 행 수와 일치."""
        data = _make_single_data(20)
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=1000)
        received: list[AnyEvent] = []

        async def handler(event: AnyEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        assert len(received) == 20
        assert feed.bars_emitted == 20

    async def test_bar_fields_match_dataframe(self) -> None:
        """BarEvent 필드가 DataFrame 값과 일치."""
        data = _make_single_data(5)
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=100)
        received: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            received.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        df = data.ohlcv
        for i, bar in enumerate(received):
            row = df.iloc[i]
            assert bar.symbol == "BTC/USDT"
            assert bar.timeframe == "1D"
            assert abs(bar.open - float(row["open"])) < 1e-6
            assert abs(bar.close - float(row["close"])) < 1e-6

    async def test_timestamps_monotonically_increasing(self) -> None:
        """bar_timestamp가 단조 증가."""
        data = _make_single_data(15)
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=100)
        timestamps: list[datetime] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            timestamps.append(event.bar_timestamp)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    async def test_each_bar_has_unique_correlation_id(self) -> None:
        """각 bar는 고유한 correlation_id."""
        data = _make_single_data(10)
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=100)
        cids: list[object] = []

        async def handler(event: AnyEvent) -> None:
            cids.append(event.correlation_id)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        assert len(set(cids)) == 10  # 모두 고유


class TestHistoricalDataFeedMulti:
    """멀티 심볼 리플레이 테스트."""

    async def test_multi_symbol_bar_count(self) -> None:
        """멀티 심볼: 총 BarEvent = bars x symbols."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        data = _make_multi_data(symbols, n=10)
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=1000)
        received: list[AnyEvent] = []

        async def handler(event: AnyEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        assert len(received) == 30  # 10 bars x 3 symbols
        assert feed.bars_emitted == 30

    async def test_same_timestamp_same_correlation_id(self) -> None:
        """동일 타임스탬프의 심볼들은 같은 correlation_id를 공유."""
        symbols = ["BTC/USDT", "ETH/USDT"]
        data = _make_multi_data(symbols, n=5)
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=100)
        received: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            received.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        # 10개 (5 bars x 2 symbols)
        assert len(received) == 10

        # 연속 2개씩 같은 correlation_id
        for i in range(0, len(received), 2):
            assert received[i].correlation_id == received[i + 1].correlation_id
            assert received[i].symbol != received[i + 1].symbol

    async def test_multi_symbol_ordering(self) -> None:
        """멀티 심볼: 타임스탬프 순서로 심볼 그룹 발행."""
        symbols = ["BTC/USDT", "ETH/USDT"]
        data = _make_multi_data(symbols, n=3)
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=100)
        received: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            received.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        # BTC, ETH, BTC, ETH, BTC, ETH 순서
        expected_symbols = ["BTC/USDT", "ETH/USDT"] * 3
        actual_symbols = [b.symbol for b in received]
        assert actual_symbols == expected_symbols


class TestHistoricalDataFeedEdgeCases:
    """엣지 케이스 테스트."""

    async def test_empty_dataframe(self) -> None:
        """빈 DataFrame: 0개 BarEvent."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = start + timedelta(days=1)
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz=UTC),
        )
        data = MarketDataSet(
            symbol="BTC/USDT",
            timeframe="1D",
            start=start,
            end=end,
            ohlcv=empty_df,
        )
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=100)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        assert feed.bars_emitted == 0

    async def test_heartbeat_emission(self) -> None:
        """heartbeat_interval 설정 시 HeartbeatEvent 발행."""
        data = _make_single_data(10)
        feed = HistoricalDataFeed(data, heartbeat_interval=5)
        bus = EventBus(queue_size=100)
        heartbeats: list[AnyEvent] = []

        async def handler(event: AnyEvent) -> None:
            heartbeats.append(event)

        bus.subscribe(EventType.HEARTBEAT, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        assert len(heartbeats) == 2  # bar 5, bar 10


class TestDataFeedValidation:
    """M-004: 데이터 품질 검증 테스트."""

    async def test_nan_bar_skipped(self) -> None:
        """NaN 값이 포함된 bar는 스킵."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        timestamps = pd.date_range(start=start, periods=5, freq="1D", tz=UTC)
        close = np.array([50000.0, 50100.0, float("nan"), 50300.0, 50400.0])
        df = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": [1000.0] * 5,
            },
            index=timestamps,
        )
        data = MarketDataSet(
            symbol="BTC/USDT",
            timeframe="1D",
            start=df.index[0].to_pydatetime(),  # type: ignore[union-attr]
            end=df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
            ohlcv=df,
        )
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=100)
        received: list[AnyEvent] = []

        async def handler(event: AnyEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        # NaN bar 1개 스킵 → 4개만 발행
        assert len(received) == 4
        assert feed.bars_emitted == 4

    async def test_high_less_than_low_skipped(self) -> None:
        """high < low인 bar는 스킵."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        timestamps = pd.date_range(start=start, periods=3, freq="1D", tz=UTC)
        df = pd.DataFrame(
            {
                "open": [50000.0, 50100.0, 50200.0],
                "high": [51000.0, 49000.0, 51200.0],  # 2번째: high < low
                "low": [49000.0, 50000.0, 49200.0],
                "close": [50500.0, 50050.0, 50700.0],
                "volume": [1000.0] * 3,
            },
            index=timestamps,
        )
        data = MarketDataSet(
            symbol="BTC/USDT",
            timeframe="1D",
            start=df.index[0].to_pydatetime(),  # type: ignore[union-attr]
            end=df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
            ohlcv=df,
        )
        feed = HistoricalDataFeed(data)
        bus = EventBus(queue_size=100)
        received: list[AnyEvent] = []

        async def handler(event: AnyEvent) -> None:
            received.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        assert len(received) == 2  # high < low 1개 스킵
        assert feed.bars_emitted == 2
