"""LiveDataFeed — WebSocket 실시간 1m 스트림 + CandleAggregator 연동.

CCXT Pro watch_ohlcv()를 사용하여 실시간 1m 캔들을 수신하고,
CandleAggregator로 target TF candle을 집계합니다.
심볼별 staleness detection으로 stale data 거래를 방지합니다.

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
    - Staleness Detection: 심볼별 heartbeat 타임스탬프 추적
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from src.core.events import BarEvent, RiskAlertEvent
from src.eda.candle_aggregator import CandleAggregator
from src.eda.data_feed import validate_bar

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.exchange.binance_client import BinanceClient
    from src.monitoring.metrics import WsStatusCallback

# Reconnection 상수
_INITIAL_RECONNECT_DELAY = 1.0
_MAX_RECONNECT_DELAY = 60.0

# Staleness detection: 마지막 수신 후 이 시간(초) 초과 시 stale 경고
_DEFAULT_STALENESS_TIMEOUT = 120.0  # 2분 (1m 캔들 2개 미수신)


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
        staleness_timeout: float = _DEFAULT_STALENESS_TIMEOUT,
        ws_status_callback: WsStatusCallback | None = None,
    ) -> None:
        self._symbols = symbols
        self._target_tf = target_timeframe
        self._client = client
        self._aggregator = CandleAggregator(target_timeframe)
        self._bars_emitted: int = 0
        self._shutdown_event = asyncio.Event()
        # Staleness detection
        self._staleness_timeout = staleness_timeout
        self._last_received: dict[str, float] = {}  # symbol → monotonic timestamp
        self._stale_symbols: set[str] = set()
        # WS 상태 콜백 (선택적)
        self._ws_callback = ws_status_callback
        # EventBus 참조 (stale alert 발행용, start() 시 설정)
        self._bus: EventBus | None = None

    async def start(self, bus: EventBus) -> None:
        """심볼별 WebSocket 스트림 시작. stop() 호출까지 실행."""
        import time

        self._bus = bus

        # 초기 heartbeat 설정 + WS 연결 상태 초기화
        now = time.monotonic()
        for sym in self._symbols:
            self._last_received[sym] = now
            if self._ws_callback is not None:
                self._ws_callback.on_ws_status(sym, connected=True)

        stream_tasks = [
            asyncio.create_task(self._stream_symbol(sym, bus, stagger_delay=i * 0.5))
            for i, sym in enumerate(self._symbols)
        ]
        staleness_task = asyncio.create_task(self._staleness_monitor())

        # shutdown_event 대기 task — stop() 호출 시 stream tasks를 cancel
        async def _wait_shutdown() -> None:
            await self._shutdown_event.wait()
            for t in stream_tasks:
                t.cancel()
            staleness_task.cancel()

        shutdown_task = asyncio.create_task(_wait_shutdown())

        try:
            await asyncio.gather(*stream_tasks, return_exceptions=True)
        finally:
            shutdown_task.cancel()
            staleness_task.cancel()
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

    @property
    def stale_symbols(self) -> set[str]:
        """현재 stale 상태인 심볼."""
        return self._stale_symbols

    def set_ws_callback(self, callback: WsStatusCallback) -> None:
        """WS 상태 콜백 설정.

        Args:
            callback: WsStatusCallback 구현체
        """
        self._ws_callback = callback

    def _record_heartbeat(self, symbol: str) -> None:
        """심볼 데이터 수신 시 heartbeat 기록."""
        import time

        self._last_received[symbol] = time.monotonic()
        if symbol in self._stale_symbols:
            self._stale_symbols.discard(symbol)
            logger.info("{} data feed recovered from stale state", symbol)

    async def _staleness_monitor(self) -> None:
        """주기적으로 심볼별 데이터 수신 시간을 확인하여 stale 경고."""
        import time

        _check_interval = min(self._staleness_timeout / 2, 30.0)
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=_check_interval)
                break  # shutdown event set
            except TimeoutError:
                pass  # timeout → staleness check 수행

            now = time.monotonic()
            for sym in self._symbols:
                last = self._last_received.get(sym, now)
                elapsed = now - last
                if elapsed > self._staleness_timeout and sym not in self._stale_symbols:
                    self._stale_symbols.add(sym)
                    logger.critical(
                        "STALE DATA: {} no data for {:.0f}s (threshold={:.0f}s). Trading on outdated prices!",
                        sym,
                        elapsed,
                        self._staleness_timeout,
                    )
                    # RiskAlertEvent → NotificationEngine → Discord ALERTS
                    if self._bus is not None:
                        alert = RiskAlertEvent(
                            alert_level="WARNING",
                            message=f"STALE DATA: {sym} no data for {elapsed:.0f}s",
                            source="LiveDataFeed",
                        )
                        await self._bus.publish(alert)

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
                    if self._ws_callback is not None:
                        self._ws_callback.on_ws_status(symbol, connected=True)

                if not ohlcvs:
                    continue

                self._record_heartbeat(symbol)
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
                if not was_disconnected and self._ws_callback is not None:
                    self._ws_callback.on_ws_status(symbol, connected=False)
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
