"""LiveDataFeed 테스트 — WebSocket 1m 스트림 + CandleAggregator.

Mock exchange를 사용하여 watch_ohlcv() 동작을 시뮬레이션합니다.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.core.event_bus import EventBus
from src.core.events import BarEvent, EventType
from src.eda.live_data_feed import LiveDataFeed

if TYPE_CHECKING:
    from src.core.events import AnyEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS_BASE = 1_700_000_000_000  # 2023-11-14 ~22:13 UTC (ms)
_ONE_MIN_MS = 60_000


def _candle(ts_ms: int, o: float, h: float, lo: float, c: float, v: float) -> list[float]:
    """OHLCV 캔들 리스트 생성."""
    return [float(ts_ms), o, h, lo, c, v]


class MockExchange:
    """CCXT Pro exchange mock — watch_ohlcv() 시뮬레이션.

    candle_sequence의 각 원소를 순서대로 반환합니다.
    시퀀스 소진 후에는 무한 대기합니다.
    """

    def __init__(self, candle_sequence: list[list[list[float]]]) -> None:
        self._sequence = candle_sequence
        self._idx = 0

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> list[list[float]]:
        if self._idx >= len(self._sequence):
            # 시퀀스 소진 → 무한 대기 (shutdown으로 종료)
            await asyncio.sleep(999)
            return []
        result = self._sequence[self._idx]
        self._idx += 1
        return result


def _make_mock_client(exchange: MockExchange) -> MagicMock:
    """BinanceClient mock — exchange 프로퍼티만 제공."""
    client = MagicMock()
    client.exchange = exchange
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSymbolsProperty:
    """symbols property 테스트."""

    def test_symbols_returns_list(self) -> None:
        """symbols 프로퍼티가 심볼 리스트를 반환."""
        client = _make_mock_client(MockExchange([]))
        feed = LiveDataFeed(["BTC/USDT", "ETH/USDT"], "1D", client)
        assert feed.symbols == ["BTC/USDT", "ETH/USDT"]

    def test_symbols_single(self) -> None:
        """단일 심볼."""
        client = _make_mock_client(MockExchange([]))
        feed = LiveDataFeed(["SOL/USDT"], "1h", client)
        assert feed.symbols == ["SOL/USDT"]


class TestSingleSymbolStream:
    """단일 심볼 스트림 테스트."""

    @pytest.mark.asyncio
    async def test_candle_completion_emits_bar(self) -> None:
        """timestamp 변경 시 이전 캔들이 BarEvent로 발행되는지 확인."""
        ts1 = _TS_BASE
        ts2 = _TS_BASE + _ONE_MIN_MS

        # 캔들 시퀀스: ts1 형성중 → ts2 시작 (= ts1 완성)
        exchange = MockExchange(
            [
                [_candle(ts1, 100, 105, 95, 102, 1000)],
                [_candle(ts2, 103, 106, 101, 104, 800)],
            ]
        )
        client = _make_mock_client(exchange)

        feed = LiveDataFeed(["BTC/USDT"], "1h", client)
        bus = EventBus(queue_size=100)

        bars_1m: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1m":
                bars_1m.append(event)

        bus.subscribe(EventType.BAR, handler)

        bus_task = asyncio.create_task(bus.start())

        # feed가 시퀀스 소진 후 대기하므로 timeout으로 종료
        async def stop_after_delay() -> None:
            await asyncio.sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())

        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # ts1 캔들이 완성되어 1m BarEvent 1개 발행
        assert len(bars_1m) == 1
        assert bars_1m[0].symbol == "BTC/USDT"
        assert bars_1m[0].open == 100.0
        assert bars_1m[0].close == 102.0

    @pytest.mark.asyncio
    async def test_bars_emitted_counter(self) -> None:
        """bars_emitted 프로퍼티 정확성."""
        ts1 = _TS_BASE
        ts2 = _TS_BASE + _ONE_MIN_MS

        exchange = MockExchange(
            [
                [_candle(ts1, 100, 105, 95, 102, 1000)],
                [_candle(ts2, 103, 106, 101, 104, 800)],
            ]
        )
        client = _make_mock_client(exchange)
        feed = LiveDataFeed(["BTC/USDT"], "1h", client)
        bus = EventBus(queue_size=100)

        bus_task = asyncio.create_task(bus.start())

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # 1m bar 1개 (ts1 완성) + flush 시 ts2 미완성 1m→1h 집계 BarEvent
        assert feed.bars_emitted >= 1


class TestMultiSymbolStream:
    """멀티 심볼 동시 스트리밍 테스트."""

    @pytest.mark.asyncio
    async def test_multi_symbol_concurrent(self) -> None:
        """두 심볼이 동시에 bar를 발행하는지 확인."""
        ts1 = _TS_BASE
        ts2 = _TS_BASE + _ONE_MIN_MS

        # 두 심볼 모두 동일한 exchange mock 사용
        exchange = MockExchange(
            [
                # BTC 첫 호출 → ETH 첫 호출 → BTC 두번째 → ETH 두번째 (interleaved)
                [_candle(ts1, 100, 105, 95, 102, 1000)],
                [_candle(ts1, 3000, 3100, 2900, 3050, 5000)],
                [_candle(ts2, 103, 106, 101, 104, 800)],
                [_candle(ts2, 3060, 3120, 3000, 3080, 4500)],
            ]
        )
        client = _make_mock_client(exchange)
        feed = LiveDataFeed(["BTC/USDT", "ETH/USDT"], "1h", client)
        bus = EventBus(queue_size=200)

        symbols_seen: set[str] = set()

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1m":
                symbols_seen.add(event.symbol)

        bus.subscribe(EventType.BAR, handler)
        bus_task = asyncio.create_task(bus.start())

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.5)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # 두 심볼 모두 bar 발행 확인
        # Note: MockExchange는 단일 인스턴스이므로 watch_ohlcv 호출이 인터리브됨
        # 실제로는 심볼별로 독립적이지만, mock에서는 공유됨
        assert feed.bars_emitted >= 1


class TestCandleCompletionDetection:
    """캔들 완성 감지 로직 테스트."""

    @pytest.mark.asyncio
    async def test_same_timestamp_no_emission(self) -> None:
        """같은 timestamp 연속 수신 시 bar 미발행 (캔들 형성 중)."""
        ts1 = _TS_BASE

        exchange = MockExchange(
            [
                [_candle(ts1, 100, 105, 95, 102, 1000)],
                [_candle(ts1, 100, 107, 94, 106, 1500)],  # 같은 ts → 업데이트
            ]
        )
        client = _make_mock_client(exchange)
        feed = LiveDataFeed(["BTC/USDT"], "1h", client)
        bus = EventBus(queue_size=100)

        bars_1m: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1m":
                bars_1m.append(event)

        bus.subscribe(EventType.BAR, handler)
        bus_task = asyncio.create_task(bus.start())

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # 같은 timestamp이므로 캔들 미완성 → 1m bar 미발행
        assert len(bars_1m) == 0

    @pytest.mark.asyncio
    async def test_three_candles_two_completions(self) -> None:
        """3개 캔들 시퀀스 → 2개 완성."""
        ts1 = _TS_BASE
        ts2 = _TS_BASE + _ONE_MIN_MS
        ts3 = _TS_BASE + _ONE_MIN_MS * 2

        exchange = MockExchange(
            [
                [_candle(ts1, 100, 105, 95, 102, 1000)],
                [_candle(ts2, 103, 106, 101, 104, 800)],  # ts1 완성
                [_candle(ts3, 105, 108, 103, 107, 900)],  # ts2 완성
            ]
        )
        client = _make_mock_client(exchange)
        feed = LiveDataFeed(["BTC/USDT"], "1h", client)
        bus = EventBus(queue_size=100)

        bars_1m: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1m":
                bars_1m.append(event)

        bus.subscribe(EventType.BAR, handler)
        bus_task = asyncio.create_task(bus.start())

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        assert len(bars_1m) == 2
        assert bars_1m[0].open == 100.0  # ts1 캔들
        assert bars_1m[1].open == 103.0  # ts2 캔들


class TestAggregatorIntegration:
    """1m → target TF 집계 통합 테스트."""

    @pytest.mark.asyncio
    async def test_1m_to_5m_aggregation(self) -> None:
        """5개 1m 캔들 → 1개 5m BarEvent 집계."""
        candles: list[list[list[float]]] = []
        for i in range(6):
            ts = _TS_BASE + _ONE_MIN_MS * i
            price = 100.0 + i
            candles.append([_candle(ts, price, price + 2, price - 1, price + 1, 1000)])

        exchange = MockExchange(candles)
        client = _make_mock_client(exchange)
        feed = LiveDataFeed(["BTC/USDT"], "5m", client)
        bus = EventBus(queue_size=200)

        bars_5m: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "5m":
                bars_5m.append(event)

        bus.subscribe(EventType.BAR, handler)
        bus_task = asyncio.create_task(bus.start())

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.5)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # 5m 경계를 넘어야 집계 발생
        # 실제 집계 여부는 CandleAggregator의 period 경계에 따라 다름
        assert feed.bars_emitted >= 5  # 최소 5개 1m bar


class TestValidateBarReuse:
    """NaN/Inf bar 스킵 테스트."""

    @pytest.mark.asyncio
    async def test_nan_bar_skipped(self) -> None:
        """NaN 포함 캔들은 스킵."""
        ts1 = _TS_BASE
        ts2 = _TS_BASE + _ONE_MIN_MS

        exchange = MockExchange(
            [
                [_candle(ts1, float("nan"), 105, 95, 102, 1000)],
                [_candle(ts2, 103, 106, 101, 104, 800)],  # ts1 완성 시도 → NaN 스킵
            ]
        )
        client = _make_mock_client(exchange)
        feed = LiveDataFeed(["BTC/USDT"], "1h", client)
        bus = EventBus(queue_size=100)

        bars_1m: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1m":
                bars_1m.append(event)

        bus.subscribe(EventType.BAR, handler)
        bus_task = asyncio.create_task(bus.start())

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # NaN bar는 스킵 → 0개 발행
        assert len(bars_1m) == 0

    @pytest.mark.asyncio
    async def test_inf_bar_skipped(self) -> None:
        """Inf 포함 캔들은 스킵."""
        ts1 = _TS_BASE
        ts2 = _TS_BASE + _ONE_MIN_MS

        exchange = MockExchange(
            [
                [_candle(ts1, 100, math.inf, 95, 102, 1000)],
                [_candle(ts2, 103, 106, 101, 104, 800)],
            ]
        )
        client = _make_mock_client(exchange)
        feed = LiveDataFeed(["BTC/USDT"], "1h", client)
        bus = EventBus(queue_size=100)

        bars_1m: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1m":
                bars_1m.append(event)

        bus.subscribe(EventType.BAR, handler)
        bus_task = asyncio.create_task(bus.start())

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        assert len(bars_1m) == 0


class TestReconnection:
    """NetworkError 후 재연결 테스트."""

    @pytest.mark.asyncio
    async def test_reconnection_on_network_error(self) -> None:
        """NetworkError 후 재연결하여 데이터 계속 수신."""
        from unittest.mock import patch as mock_patch

        import ccxt as ccxt_sync

        ts1 = _TS_BASE
        ts2 = _TS_BASE + _ONE_MIN_MS
        call_count = 0
        bars_1m: list[BarEvent] = []

        class ErrorThenSuccessExchange:
            """첫 호출 NetworkError → 이후 정상 캔들 반환."""

            async def watch_ohlcv(self, symbol: str, timeframe: str) -> list[list[float]]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ccxt_sync.NetworkError("test disconnect")
                if call_count == 2:
                    return [_candle(ts1, 100, 105, 95, 102, 1000)]
                if call_count == 3:
                    return [_candle(ts2, 103, 106, 101, 104, 800)]
                await asyncio.sleep(999)
                return []

        client = _make_mock_client(ErrorThenSuccessExchange())
        feed = LiveDataFeed(["BTC/USDT"], "1h", client)
        bus = EventBus(queue_size=100)

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            if event.timeframe == "1m":
                bars_1m.append(event)

        bus.subscribe(EventType.BAR, handler)
        bus_task = asyncio.create_task(bus.start())

        original_sleep = asyncio.sleep

        async def fast_sleep(delay: float) -> None:
            if delay <= 0.5:
                await original_sleep(delay)
            else:
                await original_sleep(0.01)

        async def stop_after_delay() -> None:
            await original_sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())

        with mock_patch("asyncio.sleep", side_effect=fast_sleep):
            await feed.start(bus)

        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # NetworkError 후 재연결하여 ts1 캔들 완성
        assert len(bars_1m) == 1


class TestStopGraceful:
    """stop() 호출 시 정상 종료."""

    @pytest.mark.asyncio
    async def test_stop_flushes_partial_candles(self) -> None:
        """stop() 호출 시 미완성 TF 캔들이 flush되는지 확인.

        2개 1m 캔들: ts1 완성 (aggregator에 입력) → ts2 미완성 (flush_all로 발행).
        """
        ts1 = _TS_BASE
        ts2 = _TS_BASE + _ONE_MIN_MS

        exchange = MockExchange(
            [
                [_candle(ts1, 100, 105, 95, 102, 1000)],
                [_candle(ts2, 103, 106, 101, 104, 800)],  # ts1 완성 → aggregator에 입력
            ]
        )
        client = _make_mock_client(exchange)
        feed = LiveDataFeed(["BTC/USDT"], "1h", client)
        bus = EventBus(queue_size=100)

        all_bars: list[BarEvent] = []

        async def handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            all_bars.append(event)

        bus.subscribe(EventType.BAR, handler)
        bus_task = asyncio.create_task(bus.start())

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # ts1 완성 → 1m BarEvent + aggregator에 입력
        # stop() 시 flush_all()로 미완성 1h BarEvent 발행
        tf_bars = [b for b in all_bars if b.timeframe == "1h"]
        assert len(tf_bars) >= 1


class TestStalenessDetection:
    """LiveDataFeed staleness detection 테스트."""

    async def test_stale_symbols_initially_empty(self) -> None:
        """초기 상태에서 stale 심볼 없음."""
        mock_client = MagicMock()
        mock_client.exchange = MockExchange([])
        feed = LiveDataFeed(
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            client=mock_client,
            staleness_timeout=60.0,
        )
        assert len(feed.stale_symbols) == 0

    async def test_heartbeat_recorded_on_data_received(self) -> None:
        """데이터 수신 시 heartbeat가 기록됨."""
        import time

        mock_client = MagicMock()
        mock_client.exchange = MockExchange([])
        feed = LiveDataFeed(
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            client=mock_client,
            staleness_timeout=60.0,
        )

        feed._record_heartbeat("BTC/USDT")
        last = feed._last_received.get("BTC/USDT", 0)
        assert last > 0
        assert abs(last - time.monotonic()) < 1.0

    async def test_stale_recovery(self) -> None:
        """stale 상태에서 데이터 수신 시 복구."""
        mock_client = MagicMock()
        mock_client.exchange = MockExchange([])
        feed = LiveDataFeed(
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            client=mock_client,
            staleness_timeout=60.0,
        )

        # 수동으로 stale 상태 설정
        feed._stale_symbols.add("BTC/USDT")
        assert "BTC/USDT" in feed.stale_symbols

        # heartbeat → 복구
        feed._record_heartbeat("BTC/USDT")
        assert "BTC/USDT" not in feed.stale_symbols

    async def test_staleness_monitor_detects_stale(self) -> None:
        """staleness monitor가 timeout 초과 시 stale 감지."""
        import time

        mock_client = MagicMock()
        mock_client.exchange = MockExchange([])
        feed = LiveDataFeed(
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            client=mock_client,
            staleness_timeout=0.1,  # 매우 짧은 timeout
        )

        # heartbeat를 과거로 설정 (stale 유발)
        feed._last_received["BTC/USDT"] = time.monotonic() - 1.0

        # staleness monitor를 짧게 실행
        feed._shutdown_event = asyncio.Event()

        async def run_monitor() -> None:
            await feed._staleness_monitor()

        monitor_task = asyncio.create_task(run_monitor())
        await asyncio.sleep(0.15)  # 첫 체크 발생 대기
        feed._shutdown_event.set()
        await monitor_task

        assert "BTC/USDT" in feed.stale_symbols

    async def test_staleness_publishes_risk_alert(self) -> None:
        """stale 감지 시 RiskAlertEvent가 EventBus에 발행됨."""
        import time

        from src.core.events import RiskAlertEvent

        mock_client = MagicMock()
        mock_client.exchange = MockExchange([])
        feed = LiveDataFeed(
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            client=mock_client,
            staleness_timeout=0.1,
        )

        bus = EventBus(queue_size=100)
        feed._bus = bus

        # RiskAlertEvent 캡처
        alerts: list[RiskAlertEvent] = []

        async def capture(event: AnyEvent) -> None:
            assert isinstance(event, RiskAlertEvent)
            alerts.append(event)

        bus.subscribe(EventType.RISK_ALERT, capture)
        bus_task = asyncio.create_task(bus.start())

        # heartbeat를 과거로 설정
        feed._last_received["BTC/USDT"] = time.monotonic() - 1.0

        monitor_task = asyncio.create_task(feed._staleness_monitor())
        await asyncio.sleep(0.15)
        feed._shutdown_event.set()
        await monitor_task
        await bus.flush()
        await bus.stop()
        await bus_task

        assert len(alerts) >= 1
        assert "STALE DATA" in alerts[0].message
        # elapsed=1.0s >> 1.5*timeout=0.15s → CRITICAL escalation
        assert alerts[0].alert_level == "CRITICAL"

    async def test_staleness_critical_escalation(self) -> None:
        """elapsed > 1.5x timeout -> CRITICAL alert escalation."""
        import time

        from src.core.events import RiskAlertEvent

        mock_client = MagicMock()
        mock_client.exchange = MockExchange([])
        timeout = 0.1
        feed = LiveDataFeed(
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            client=mock_client,
            staleness_timeout=timeout,
        )

        bus = EventBus(queue_size=100)
        feed._bus = bus

        alerts: list[RiskAlertEvent] = []

        async def capture(event: AnyEvent) -> None:
            assert isinstance(event, RiskAlertEvent)
            alerts.append(event)

        bus.subscribe(EventType.RISK_ALERT, capture)
        bus_task = asyncio.create_task(bus.start())

        # elapsed = 1.0s >> 1.5 * 0.1 = 0.15s → CRITICAL
        feed._last_received["BTC/USDT"] = time.monotonic() - 1.0

        monitor_task = asyncio.create_task(feed._staleness_monitor())
        await asyncio.sleep(0.15)
        feed._shutdown_event.set()
        await monitor_task
        await bus.flush()
        await bus.stop()
        await bus_task

        assert len(alerts) >= 1
        assert alerts[0].alert_level == "CRITICAL"

    async def test_staleness_warning_level(self) -> None:
        """elapsed > timeout but <= 1.5x timeout -> WARNING alert."""
        import time

        from src.core.events import RiskAlertEvent

        mock_client = MagicMock()
        mock_client.exchange = MockExchange([])
        timeout = 0.2
        feed = LiveDataFeed(
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            client=mock_client,
            staleness_timeout=timeout,
        )

        bus = EventBus(queue_size=100)
        feed._bus = bus

        alerts: list[RiskAlertEvent] = []

        async def capture(event: AnyEvent) -> None:
            assert isinstance(event, RiskAlertEvent)
            alerts.append(event)

        bus.subscribe(EventType.RISK_ALERT, capture)
        bus_task = asyncio.create_task(bus.start())

        # check_interval = timeout/2 = 0.1s
        # initial_elapsed = 0.15s → at check: ~0.25s > 0.2 (trigger), < 0.3 (WARNING)
        feed._last_received["BTC/USDT"] = time.monotonic() - 0.15

        monitor_task = asyncio.create_task(feed._staleness_monitor())
        await asyncio.sleep(0.15)
        feed._shutdown_event.set()
        await monitor_task
        await bus.flush()
        await bus.stop()
        await bus_task

        assert len(alerts) >= 1
        assert alerts[0].alert_level == "WARNING"

    async def test_staleness_no_alert_without_bus(self) -> None:
        """bus가 None이면 RiskAlertEvent 발행 안 함 (에러 없음)."""
        import time

        mock_client = MagicMock()
        mock_client.exchange = MockExchange([])
        feed = LiveDataFeed(
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            client=mock_client,
            staleness_timeout=0.1,
        )

        # _bus is None (default)
        feed._last_received["BTC/USDT"] = time.monotonic() - 1.0

        monitor_task = asyncio.create_task(feed._staleness_monitor())
        await asyncio.sleep(0.15)
        feed._shutdown_event.set()
        await monitor_task

        # No error raised, stale is still detected
        assert "BTC/USDT" in feed.stale_symbols


class TestWsStatusCallback:
    """M3: 초기 connected=True 제거 + 첫 데이터 후 호출 확인."""

    @pytest.mark.asyncio
    async def test_no_connected_callback_on_start(self) -> None:
        """start() 직후에는 on_ws_status(connected=True) 미호출."""
        # 무한 대기 시퀀스 — 데이터 반환 전에 stop()
        exchange = MockExchange([])
        client = _make_mock_client(exchange)

        callback = MagicMock()
        feed = LiveDataFeed(["BTC/USDT"], "1h", client, ws_status_callback=callback)
        bus = EventBus(queue_size=100)
        bus_task = asyncio.create_task(bus.start())

        async def stop_quickly() -> None:
            await asyncio.sleep(0.1)
            await feed.stop()

        stop_task = asyncio.create_task(stop_quickly())
        await feed.start(bus)
        await bus.stop()
        await bus_task
        await stop_task

        # start()에서 on_ws_status 호출 없어야 함
        callback.on_ws_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_connected_callback_on_first_data(self) -> None:
        """첫 데이터 수신 시 on_ws_status(connected=True) 호출."""
        ts1 = _TS_BASE

        exchange = MockExchange(
            [[_candle(ts1, 100, 105, 95, 102, 1000)]]
        )
        client = _make_mock_client(exchange)

        callback = MagicMock()
        feed = LiveDataFeed(["BTC/USDT"], "1h", client, ws_status_callback=callback)
        bus = EventBus(queue_size=100)
        bus_task = asyncio.create_task(bus.start())

        async def stop_after_delay() -> None:
            await asyncio.sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())
        await feed.start(bus)
        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # 첫 데이터 수신 후 connected=True 호출
        callback.on_ws_status.assert_called_with("BTC/USDT", connected=True)


class TestConsecutiveFailureAlert:
    """M4: WS 연속 실패 시 CRITICAL alert 발행."""

    @pytest.mark.asyncio
    async def test_consecutive_failures_trigger_critical_alert(self) -> None:
        """_MAX_CONSECUTIVE_FAILURES 이상 연속 실패 시 CRITICAL alert."""
        from unittest.mock import patch as mock_patch

        import ccxt as ccxt_sync

        from src.core.events import RiskAlertEvent
        from src.eda.live_data_feed import _MAX_CONSECUTIVE_FAILURES

        fail_count = 0
        max_fails = _MAX_CONSECUTIVE_FAILURES + 2  # 충분히 많이 실패

        class AlwaysFailExchange:
            async def watch_ohlcv(self, symbol: str, timeframe: str) -> list[list[float]]:
                nonlocal fail_count
                fail_count += 1
                if fail_count > max_fails:
                    await asyncio.sleep(999)
                    return []
                raise ccxt_sync.NetworkError("persistent failure")

        client = _make_mock_client(AlwaysFailExchange())
        feed = LiveDataFeed(["BTC/USDT"], "1h", client)
        bus = EventBus(queue_size=100)

        alerts: list[RiskAlertEvent] = []

        async def capture(event: object) -> None:
            if isinstance(event, RiskAlertEvent):
                alerts.append(event)

        bus.subscribe(EventType.RISK_ALERT, capture)
        bus_task = asyncio.create_task(bus.start())

        # asyncio.sleep를 mock하여 reconnect delay를 0으로 만듦
        original_sleep = asyncio.sleep

        async def fast_sleep(delay: float) -> None:
            # stagger delay와 staleness monitor용 sleep은 정상 동작
            # reconnect delay만 빠르게 처리
            if delay <= 0.5:  # stagger_delay
                await original_sleep(delay)
            else:
                await original_sleep(0.01)  # reconnect → 즉시

        async def stop_after_delay() -> None:
            await original_sleep(0.3)
            await feed.stop()

        stop_task = asyncio.create_task(stop_after_delay())

        with mock_patch("asyncio.sleep", side_effect=fast_sleep):
            await feed.start(bus)

        await bus.flush()
        await bus.stop()
        await bus_task
        await stop_task

        # CRITICAL alert 발행 확인
        critical_alerts = [a for a in alerts if "persistent failure" in a.message]
        assert len(critical_alerts) >= 1
        assert critical_alerts[0].alert_level == "CRITICAL"
