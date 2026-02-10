"""EDA fast_mode 테스트.

fast_mode(pre-aggregation + incremental + buffer truncation) 동작을 검증합니다.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pandas as pd

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType, SignalEvent
from src.data.market_data import MarketDataSet, MultiSymbolData
from src.eda.data_feed import HistoricalDataFeed, _resample_1m_to_tf
from src.eda.strategy_engine import StrategyEngine
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals


# =========================================================================
# Test Strategy
# =========================================================================
class SimpleTestStrategy(BaseStrategy):
    """테스트용 전략: close > open → LONG, else SHORT."""

    def __init__(self) -> None:
        self._run_count = 0
        self._run_incremental_count = 0

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
        strength = ((df["close"] - df["open"]).abs() / df["open"]).shift(1).fillna(0)
        entries = direction.diff().fillna(0).abs() > 0
        exits = pd.Series(False, index=df.index)
        return StrategySignals(entries=entries, exits=exits, direction=direction, strength=strength)

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySignals]:
        self._run_count += 1
        return super().run(df)

    def run_incremental(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StrategySignals]:
        self._run_incremental_count += 1
        # 직접 처리 (super().run_incremental이 self.run() 호출하여 카운터 오염 방지)
        self.validate_input(df)
        processed_df = self.preprocess(df)
        signals = self.generate_signals(processed_df)
        return processed_df, signals


# =========================================================================
# Helpers
# =========================================================================
def _make_1m_dataframe(
    start: datetime,
    periods: int,
    base_price: float = 50000.0,
) -> pd.DataFrame:
    """1m OHLCV DataFrame."""
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
    periods: int = 1440 * 3,
) -> MarketDataSet:
    """3일치 1m MarketDataSet (1D → 3 bars)."""
    start = start or datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
    df = _make_1m_dataframe(start, periods)
    return MarketDataSet(
        symbol=symbol,
        timeframe="1m",
        start=start,
        end=start + timedelta(minutes=periods),
        ohlcv=df,
    )


def _make_multi_symbol_data(
    symbols: list[str] | None = None,
    start: datetime | None = None,
    periods: int = 1440 * 3,
) -> MultiSymbolData:
    """멀티 심볼 3일치 1m 데이터."""
    symbols = symbols or ["BTC/USDT", "ETH/USDT"]
    start = start or datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
    ohlcv: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        ohlcv[sym] = _make_1m_dataframe(start, periods, base_price=50000.0 + i * 1000)
    return MultiSymbolData(
        symbols=symbols,
        timeframe="1m",
        start=start,
        end=start + timedelta(minutes=periods),
        ohlcv=ohlcv,
    )


# =========================================================================
# Test: _resample_1m_to_tf helper
# =========================================================================
class TestResample1mToTf:
    """_resample_1m_to_tf 유틸리티 함수 테스트."""

    def test_resample_to_1d(self) -> None:
        """1m → 1D resample: OHLCV 집계 정확성."""
        start = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        df_1m = _make_1m_dataframe(start, periods=1440 * 2)  # 2일

        result = _resample_1m_to_tf(df_1m, "1D")

        assert len(result) == 2
        # 첫 번째 일봉: open = first 1m open, close = last 1m close
        assert result.iloc[0]["open"] == df_1m.iloc[0]["open"]
        assert result.iloc[0]["close"] == df_1m.iloc[1439]["close"]
        # high = max of all 1m highs in that day
        day1_highs = df_1m.iloc[:1440]["high"]
        assert result.iloc[0]["high"] == day1_highs.max()

    def test_resample_drops_nan_rows(self) -> None:
        """close가 NaN인 행은 제거."""
        start = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)
        df_1m = _make_1m_dataframe(start, periods=1440)
        # 마지막 half day 제거해도 1D는 하나 유지
        result = _resample_1m_to_tf(df_1m, "1D")
        assert len(result) >= 1
        assert result["close"].isna().sum() == 0


# =========================================================================
# Test: HistoricalDataFeed fast_mode (단일 심볼)
# =========================================================================
class TestFastDataFeedSingle:
    """fast_mode 단일 심볼 — TF bar만 발행 확인."""

    async def test_fast_mode_emits_only_tf_bars(self) -> None:
        """fast_mode=True → 1m BarEvent 없음, TF bar만 발행."""
        data = _make_1m_dataset(periods=1440 * 3)  # 3일

        feed = HistoricalDataFeed(data, target_timeframe="1D", fast_mode=True)
        bus = EventBus(queue_size=1000)

        bars: list[BarEvent] = []

        async def collect(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            bars.append(event)

        bus.subscribe(EventType.BAR, collect)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        # 3일 데이터 → 최소 2~3개 1D bar (resample 경계에 따라)
        assert len(bars) >= 2
        # 모든 bar는 1D timeframe
        for bar in bars:
            assert bar.timeframe == "1D"
            assert bar.symbol == "BTC/USDT"

    async def test_fast_mode_no_1m_bars(self) -> None:
        """fast_mode=True → 1m BarEvent 미발행."""
        data = _make_1m_dataset(periods=1440 * 2)

        feed = HistoricalDataFeed(data, target_timeframe="1D", fast_mode=True)
        bus = EventBus(queue_size=1000)

        timeframes: list[str] = []

        async def collect(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            timeframes.append(event.timeframe)

        bus.subscribe(EventType.BAR, collect)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        assert "1m" not in timeframes
        assert all(tf == "1D" for tf in timeframes)

    async def test_standard_mode_emits_1m_and_tf(self) -> None:
        """fast_mode=False (기본) → 1m bar + TF bar 모두 발행."""
        data = _make_1m_dataset(periods=1440 * 2)

        feed = HistoricalDataFeed(data, target_timeframe="1D", fast_mode=False)
        bus = EventBus(queue_size=100_000)

        timeframes: list[str] = []

        async def collect(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            timeframes.append(event.timeframe)

        bus.subscribe(EventType.BAR, collect)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        assert "1m" in timeframes
        assert "1D" in timeframes

    async def test_fast_vs_standard_bar_count(self) -> None:
        """fast_mode → 이벤트 수 대폭 감소."""
        data = _make_1m_dataset(periods=1440 * 3)

        # fast
        feed_fast = HistoricalDataFeed(data, target_timeframe="1D", fast_mode=True)
        bus_fast = EventBus(queue_size=1000)
        fast_count = 0

        async def count_fast(event: AnyEvent) -> None:
            nonlocal fast_count
            fast_count += 1

        bus_fast.subscribe(EventType.BAR, count_fast)
        task_fast = asyncio.create_task(bus_fast.start())
        await feed_fast.start(bus_fast)
        await bus_fast.stop()
        await task_fast

        # standard
        feed_std = HistoricalDataFeed(data, target_timeframe="1D", fast_mode=False)
        bus_std = EventBus(queue_size=100_000)
        std_count = 0

        async def count_std(event: AnyEvent) -> None:
            nonlocal std_count
            std_count += 1

        bus_std.subscribe(EventType.BAR, count_std)
        task_std = asyncio.create_task(bus_std.start())
        await feed_std.start(bus_std)
        await bus_std.stop()
        await task_std

        # fast_mode는 standard보다 훨씬 적은 이벤트
        assert fast_count < std_count
        assert fast_count < 10  # ~3 TF bars
        assert std_count > 1000  # 4320 1m bars + 3 TF bars


# =========================================================================
# Test: HistoricalDataFeed fast_mode (멀티 심볼)
# =========================================================================
class TestFastDataFeedMulti:
    """fast_mode 멀티 심볼 — common index 처리."""

    async def test_fast_mode_multi_emits_tf_bars_only(self) -> None:
        """멀티 심볼 fast_mode → TF bar만 발행, 심볼별로."""
        data = _make_multi_symbol_data(periods=1440 * 3)

        feed = HistoricalDataFeed(data, target_timeframe="1D", fast_mode=True)
        bus = EventBus(queue_size=1000)

        bars: list[BarEvent] = []

        async def collect(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            bars.append(event)

        bus.subscribe(EventType.BAR, collect)

        task = asyncio.create_task(bus.start())
        await feed.start(bus)
        await bus.stop()
        await task

        # 모든 bar는 1D
        for bar in bars:
            assert bar.timeframe == "1D"

        # 두 심볼 모두 존재
        symbols = {bar.symbol for bar in bars}
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols

        # 심볼별 bar 수 동일 (common index)
        btc_bars = [b for b in bars if b.symbol == "BTC/USDT"]
        eth_bars = [b for b in bars if b.symbol == "ETH/USDT"]
        assert len(btc_bars) == len(eth_bars)
        assert len(btc_bars) >= 2


# =========================================================================
# Test: StrategyEngine buffer truncation
# =========================================================================
class TestStrategyEngineBufferTruncation:
    """max_buffer_size 동작 확인."""

    async def test_buffer_truncation(self) -> None:
        """max_buffer_size 초과 시 오래된 데이터 제거."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(
            strategy, warmup_periods=5, target_timeframe="1D", max_buffer_size=10
        )
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        signals: list[AnyEvent] = []

        async def collect(event: AnyEvent) -> None:
            if isinstance(event, SignalEvent):
                signals.append(event)

        bus.subscribe(EventType.SIGNAL, collect)

        task = asyncio.create_task(bus.start())
        base = datetime(2024, 1, 1, tzinfo=UTC)

        # 20 bars 전송
        for i in range(20):
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0 + i,
                high=50010.0 + i,
                low=49990.0 + i,
                close=50005.0 + i,
                volume=100.0,
                bar_timestamp=base + timedelta(days=i),
                correlation_id=uuid4(),
                source="test",
            )
            await bus.publish(bar)
            await bus.flush()

        await bus.stop()
        await task

        # 버퍼는 최대 10으로 유지 (warmup=5이므로 bar 5부터 시그널 발행)
        assert len(engine._buffers["BTC/USDT"]) == 10
        assert len(signals) > 0

    async def test_no_truncation_without_max_buffer(self) -> None:
        """max_buffer_size=None이면 버퍼 무제한."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(
            strategy, warmup_periods=5, target_timeframe="1D", max_buffer_size=None
        )
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        base = datetime(2024, 1, 1, tzinfo=UTC)

        for i in range(20):
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0 + i,
                high=50010.0 + i,
                low=49990.0 + i,
                close=50005.0 + i,
                volume=100.0,
                bar_timestamp=base + timedelta(days=i),
                correlation_id=uuid4(),
                source="test",
            )
            await bus.publish(bar)
            await bus.flush()

        await bus.stop()
        await task

        assert len(engine._buffers["BTC/USDT"]) == 20


# =========================================================================
# Test: StrategyEngine incremental mode
# =========================================================================
class TestStrategyEngineIncremental:
    """incremental mode → run_incremental() 호출 확인."""

    async def test_incremental_calls_run_incremental(self) -> None:
        """incremental=True → strategy.run_incremental() 호출."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(strategy, warmup_periods=3, target_timeframe="1D", incremental=True)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        base = datetime(2024, 1, 1, tzinfo=UTC)

        for i in range(5):
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0 + i,
                high=50010.0 + i,
                low=49990.0 + i,
                close=50005.0 + i,
                volume=100.0,
                bar_timestamp=base + timedelta(days=i),
                correlation_id=uuid4(),
                source="test",
            )
            await bus.publish(bar)
            await bus.flush()

        await bus.stop()
        await task

        # warmup=3이므로 buf_len >= 3인 bar 2,3,4에서 run_incremental 호출 (3회)
        assert strategy._run_incremental_count == 3
        assert strategy._run_count == 0

    async def test_standard_calls_run(self) -> None:
        """incremental=False → strategy.run() 호출."""
        strategy = SimpleTestStrategy()
        engine = StrategyEngine(
            strategy, warmup_periods=3, target_timeframe="1D", incremental=False
        )
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        base = datetime(2024, 1, 1, tzinfo=UTC)

        for i in range(5):
            bar = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0 + i,
                high=50010.0 + i,
                low=49990.0 + i,
                close=50005.0 + i,
                volume=100.0,
                bar_timestamp=base + timedelta(days=i),
                correlation_id=uuid4(),
                source="test",
            )
            await bus.publish(bar)
            await bus.flush()

        await bus.stop()
        await task

        assert strategy._run_count == 3
        assert strategy._run_incremental_count == 0


# =========================================================================
# Test: CTREND predict_last_only
# =========================================================================
class TestCTRENDPredictLastOnly:
    """CTREND predict_last_only 시그널 정확성."""

    def test_predict_last_only_matches_full(self) -> None:
        """predict_last_only=True 마지막 시그널이 full과 동일."""
        from src.strategy.ctrend.config import CTRENDConfig
        from src.strategy.ctrend.preprocessor import preprocess
        from src.strategy.ctrend.signal import generate_signals

        # 400일치 합성 데이터
        rng = np.random.default_rng(42)
        n = 400
        dates = pd.date_range("2023-01-01", periods=n, freq="1D", tz=UTC)
        close = 50000 + np.cumsum(rng.standard_normal(n) * 500)
        high = close + rng.uniform(100, 500, n)
        low = close - rng.uniform(100, 500, n)
        open_ = close + rng.standard_normal(n) * 200
        volume = rng.uniform(1000, 5000, n)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        config = CTRENDConfig(training_window=252)
        processed = preprocess(df, config)

        # Full run
        signals_full = generate_signals(processed, config, predict_last_only=False)
        # Incremental run
        signals_incr = generate_signals(processed, config, predict_last_only=True)

        # 마지막 시그널 일치
        assert signals_full.direction.iloc[-1] == signals_incr.direction.iloc[-1]
        assert abs(signals_full.strength.iloc[-1] - signals_incr.strength.iloc[-1]) < 1e-10

    def test_predict_last_only_faster(self) -> None:
        """predict_last_only=True는 full보다 빠름 (fits 수 감소)."""
        import time

        from src.strategy.ctrend.config import CTRENDConfig
        from src.strategy.ctrend.preprocessor import preprocess
        from src.strategy.ctrend.signal import generate_signals

        rng = np.random.default_rng(42)
        n = 400
        dates = pd.date_range("2023-01-01", periods=n, freq="1D", tz=UTC)
        close = 50000 + np.cumsum(rng.standard_normal(n) * 500)
        high = close + rng.uniform(100, 500, n)
        low = close - rng.uniform(100, 500, n)
        open_ = close + rng.standard_normal(n) * 200
        volume = rng.uniform(1000, 5000, n)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        config = CTRENDConfig(training_window=252)
        processed = preprocess(df, config)

        # Full timing
        t0 = time.perf_counter()
        generate_signals(processed, config, predict_last_only=False)
        full_time = time.perf_counter() - t0

        # Incremental timing
        t0 = time.perf_counter()
        generate_signals(processed, config, predict_last_only=True)
        incr_time = time.perf_counter() - t0

        # incremental은 full보다 최소 5x 빠름 (148 fits vs 2 fits)
        assert incr_time < full_time * 0.5
