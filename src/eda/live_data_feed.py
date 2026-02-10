"""LiveDataFeed — WebSocket 실시간 1m 스트림 + CandleAggregator 연동.

CCXT Pro watch_ohlcv()를 사용하여 실시간 1m 캔들을 수신하고,
CandleAggregator로 target TF candle을 집계합니다.

데이터 흐름:
    watch_ohlcv(symbol, "1m")
        ↓ (per-symbol asyncio task)
    candle completion 감지 (timestamp 변경)
        ↓
    BarEvent(tf="1m") 발행 → PM intrabar SL/TS
        ↓
    CandleAggregator.on_1m_bar()
        ↓ (TF candle 완성 시)
    BarEvent(tf=target_tf) 발행 → Strategy + PM(full)
        ↓
    await bus.flush() — 이벤트 체인 완료 보장

Rules Applied:
    - DataFeedPort: structural subtyping 만족
    - CandleAggregator 재사용: 1m → target TF 집계
    - validate_bar 재사용: OHLCV 데이터 품질 검증
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from src.core.events import BarEvent
from src.eda.candle_aggregator import CandleAggregator
from src.eda.data_feed import validate_bar

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.exchange.binance_client import BinanceClient

# Reconnection 상수
_INITIAL_RECONNECT_DELAY = 1.0
_MAX_RECONNECT_DELAY = 60.0


class LiveDataFeed:
    """실시간 WebSocket 데이터 피드 — DataFeedPort 구현.

    CCXT Pro watch_ohlcv()로 심볼별 1m 캔들을 스트리밍하고,
    CandleAggregator로 target TF candle을 집계합니다.

    Args:
        symbols: 스트리밍할 심볼 리스트
        target_timeframe: 집계 목표 TF ("1D", "4h", "1h" 등)
        client: BinanceClient 인스턴스 (수명 관리는 LiveRunner)
    """

    def __init__(
        self,
        symbols: list[str],
        target_timeframe: str,
        client: BinanceClient,
    ) -> None:
        self._symbols = symbols
        self._target_tf = target_timeframe
        self._client = client
        self._aggregator = CandleAggregator(target_timeframe)
        self._bars_emitted: int = 0
        self._shutdown_event = asyncio.Event()

    async def start(self, bus: EventBus) -> None:
        """심볼별 WebSocket 스트림 시작. stop() 호출까지 실행."""
        stream_tasks = [
            asyncio.create_task(self._stream_symbol(sym, bus, stagger_delay=i * 0.5))
            for i, sym in enumerate(self._symbols)
        ]

        # shutdown_event 대기 task — stop() 호출 시 stream tasks를 cancel
        async def _wait_shutdown() -> None:
            await self._shutdown_event.wait()
            for t in stream_tasks:
                t.cancel()

        shutdown_task = asyncio.create_task(_wait_shutdown())

        try:
            await asyncio.gather(*stream_tasks, return_exceptions=True)
        finally:
            shutdown_task.cancel()
            # 미완성 캔들 flush
            for completed in self._aggregator.flush_all():
                await bus.publish(completed)
                self._bars_emitted += 1

    async def stop(self) -> None:
        """데이터 피드 중지."""
        self._shutdown_event.set()

    @property
    def symbols(self) -> list[str]:
        """스트리밍 대상 심볼 리스트."""
        return self._symbols

    @property
    def bars_emitted(self) -> int:
        """발행된 총 BarEvent 수."""
        return self._bars_emitted

    async def _stream_symbol(
        self, symbol: str, bus: EventBus, *, stagger_delay: float = 0.0
    ) -> None:
        """단일 심볼 WebSocket 스트림.

        CCXT Pro watch_ohlcv()는 현재 형성 중인 캔들을 반환합니다.
        timestamp 변경 = 새 캔들 시작 = 이전 캔들 완성으로 판단합니다.

        Args:
            symbol: 거래 심볼
            bus: EventBus 인스턴스
            stagger_delay: 초기 연결 지연 (동시 연결 1008 방지)
        """
        import ccxt as ccxt_sync

        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)

        exchange = self._client.exchange
        last_candle_ts: int | None = None
        prev_candle: list[float] | None = None
        reconnect_delay = _INITIAL_RECONNECT_DELAY
        was_disconnected = False

        while not self._shutdown_event.is_set():
            try:
                ohlcvs: list[list[float]] = await exchange.watch_ohlcv(symbol, "1m")
                reconnect_delay = _INITIAL_RECONNECT_DELAY  # 성공 시 리셋
                if was_disconnected:
                    logger.info("{} WebSocket reconnected", symbol)
                    was_disconnected = False

                if not ohlcvs:
                    continue

                latest = ohlcvs[-1]
                current_ts = int(latest[0])

                # 새 캔들 시작 → 이전 캔들 완성
                if (
                    last_candle_ts is not None
                    and current_ts != last_candle_ts
                    and prev_candle is not None
                ):
                    await self._emit_completed_bar(symbol, prev_candle, bus)

                prev_candle = latest
                last_candle_ts = current_ts

            except (ccxt_sync.NetworkError, OSError) as exc:
                was_disconnected = True
                logger.warning(
                    "{} WebSocket disconnected ({}), reconnecting in {:.0f}s",
                    symbol,
                    exc,
                    reconnect_delay,
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, _MAX_RECONNECT_DELAY)

    async def _emit_completed_bar(self, symbol: str, candle: list[float], bus: EventBus) -> None:
        """완성된 1m 캔들 → BarEvent 발행 + CandleAggregator 집계.

        Args:
            symbol: 거래 심볼
            candle: [timestamp_ms, open, high, low, close, volume]
            bus: EventBus 인스턴스
        """
        ts_ms = int(candle[0])
        o, h, lo, c, v = (
            float(candle[1]),
            float(candle[2]),
            float(candle[3]),
            float(candle[4]),
            float(candle[5]),
        )

        if not validate_bar(o, h, lo, c, v, symbol, ts_ms):
            return

        bar_1m = BarEvent(
            symbol=symbol,
            timeframe="1m",
            open=o,
            high=h,
            low=lo,
            close=c,
            volume=v,
            bar_timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
            correlation_id=uuid4(),
            source="LiveDataFeed",
        )
        await bus.publish(bar_1m)
        self._bars_emitted += 1

        # CandleAggregator 집계
        completed = self._aggregator.on_1m_bar(bar_1m)
        if completed is not None:
            await bus.publish(completed)
            self._bars_emitted += 1
            await bus.flush()  # TF bar 완성 시 이벤트 체인 보장
