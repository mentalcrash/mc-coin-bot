"""AggregatingDataFeed 통합 테스트.

1m 데이터를 CandleAggregator로 집계하여 TF bar를 생성하는 파이프라인을 검증합니다.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType
from src.data.market_data import MarketDataSet
from src.eda.data_feed import AggregatingDataFeed


# =========================================================================
# Helpers
# =========================================================================
def _make_1m_dataframe(
    start: datetime,
    periods: int,
    base_price: float = 50000.0,
) -> pd.DataFrame:
    """1m OHLCV DataFrame 생성.

    부드러운 가격 시리즈 (random walk 대신 deterministic).
    """
    index = pd.date_range(start=start, periods=periods, freq="1min", tz=UTC)
    data = {
        "open": [base_price + i * 0.1 for i in range(periods)],
        "high": [base_price + i * 0.1 + 5.0 for i in range(periods)],
        "low": [base_price + i * 0.1 - 5.0 for i in range(periods)],
        "close": [base_price + (i + 1) * 0.1 for i in range(periods)],
        "volume": [10.0] * periods,
    }
    return pd.DataFrame(data, index=index)


def _make_1m_dataset(
    symbol: str = "BTC/USDT",
    start: datetime | None = None,
    periods: int = 120,
) -> MarketDataSet:
    """1m MarketDataSet 생성."""
    start = start or datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
    df = _make_1m_dataframe(start, periods)
    return MarketDataSet(
        symbol=symbol,
        timeframe="1m",
        start=start,
        end=start + timedelta(minutes=periods),
        ohlcv=df,
    )


# =========================================================================
# 단일 심볼 테스트
# =========================================================================
class TestAggregatingFeedSingle:
    """단일 심볼 AggregatingDataFeed 테스트."""

    async def test_emits_1m_and_tf_bars(self) -> None:
        """1m bar와 완성된 TF bar를 모두 발행."""
        data = _make_1m_dataset(periods=120)  # 2시간 = 2개 1h candle
        feed = AggregatingDataFeed(data, target_timeframe="1h")
        bus = EventBus(queue_size=10000)

        bars_1m: list[BarEvent] = []
        bars_tf: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1m":
                bars_1m.append(event)
            elif event.timeframe == "1h":
                bars_tf.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        # 120 1m bars
        assert len(bars_1m) == 120
        # 2시간 → 2 1h candles (1개 완성 by 기간전환 + 1개 flush)
        assert len(bars_tf) == 2

    async def test_tf_bar_ohlcv_correctness(self) -> None:
        """집계된 TF bar의 OHLCV 정합성."""
        start = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        data = _make_1m_dataset(start=start, periods=120)
        feed = AggregatingDataFeed(data, target_timeframe="1h")
        bus = EventBus(queue_size=10000)

        bars_tf: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1h":
                bars_tf.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        assert len(bars_tf) >= 1
        candle = bars_tf[0]
        assert candle.symbol == "BTC/USDT"
        assert candle.timeframe == "1h"

        # OHLCV 정합성: O=first, H=max, L=min, C=last, V=sum
        df = data.ohlcv
        first_hour = df.iloc[:60]  # 12:00-12:59
        assert candle.open == first_hour["open"].iloc[0]
        assert candle.high == first_hour["high"].max()
        assert candle.low == first_hour["low"].min()
        assert candle.close == first_hour["close"].iloc[-1]
        assert abs(candle.volume - first_hour["volume"].sum()) < 0.01

    async def test_bars_emitted_count(self) -> None:
        """bars_emitted 카운터 정확성."""
        data = _make_1m_dataset(periods=60)  # 1시간
        feed = AggregatingDataFeed(data, target_timeframe="1h")
        bus = EventBus(queue_size=10000)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        # 60 1m bars + 1 flush TF bar = 61
        assert feed.bars_emitted == 61


# =========================================================================
# 1D 집계 테스트
# =========================================================================
class TestAggregatingFeed1D:
    """1m → 1D 집계."""

    async def test_1440_1m_bars_to_1d(self) -> None:
        """1440 1m bars → 1 daily candle (flush)."""
        start = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        data = _make_1m_dataset(start=start, periods=1440)
        feed = AggregatingDataFeed(data, target_timeframe="1D")
        bus = EventBus(queue_size=100000)

        bars_tf: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1D":
                bars_tf.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        # 1440분 = 정확히 1일 → flush 시 1개 candle
        assert len(bars_tf) == 1
        assert abs(bars_tf[0].volume - 14400.0) < 1.0  # 1440 * 10


# =========================================================================
# 집계 parity 테스트 (pandas resample vs CandleAggregator)
# =========================================================================
class TestAggregationParity:
    """CandleAggregator 집계 결과 vs pandas resample 결과 비교."""

    async def test_parity_with_pandas_resample(self) -> None:
        """CandleAggregator가 생성한 1h candle이 pandas resample 결과와 일치."""
        start = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        periods = 180  # 3시간
        data = _make_1m_dataset(start=start, periods=periods)
        feed = AggregatingDataFeed(data, target_timeframe="1h")
        bus = EventBus(queue_size=10000)

        bars_tf: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1h":
                bars_tf.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        # pandas resample 결과
        df = data.ohlcv
        resampled = df.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        resampled = resampled.dropna()

        assert len(bars_tf) == len(resampled)
        for bar, (_, row) in zip(bars_tf, resampled.iterrows(), strict=True):
            assert abs(bar.open - row["open"]) < 0.01
            assert abs(bar.high - row["high"]) < 0.01
            assert abs(bar.low - row["low"]) < 0.01
            assert abs(bar.close - row["close"]) < 0.01
            assert abs(bar.volume - row["volume"]) < 0.01


# =========================================================================
# Source 확인
# =========================================================================
class TestAggregatingFeedSource:
    """소스 태깅 테스트."""

    async def test_1m_bar_source(self) -> None:
        """1m bar는 AggregatingDataFeed 소스."""
        data = _make_1m_dataset(periods=5)
        feed = AggregatingDataFeed(data, target_timeframe="1h")
        bus = EventBus(queue_size=100)

        bars: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            bars.append(event)

        bus.subscribe(EventType.BAR, handler)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        one_m_bars = [b for b in bars if b.timeframe == "1m"]
        assert all(b.source == "AggregatingDataFeed" for b in one_m_bars)

        tf_bars = [b for b in bars if b.timeframe == "1h"]
        assert all(b.source == "CandleAggregator" for b in tf_bars)
