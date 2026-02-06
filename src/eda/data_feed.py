"""데이터 피드 — 히스토리컬/실시간 데이터를 BarEvent로 변환.

HistoricalDataFeed: Silver Parquet 데이터를 bar-by-bar BarEvent로 리플레이합니다.
멀티 심볼 데이터의 경우 동일 타임스탬프에 N개 BarEvent를 발행합니다.

Rules Applied:
    - #12 Data Engineering: UTC, DatetimeIndex
    - #10 Python Standards: Protocol for interface
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from src.core.events import BarEvent, EventType, HeartbeatEvent

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.data.market_data import MarketDataSet, MultiSymbolData


class HistoricalDataFeed:
    """Silver 데이터를 bar-by-bar BarEvent로 리플레이.

    단일 심볼: DataFrame row → BarEvent 순차 발행
    멀티 심볼: 동일 timestamp에 N개 BarEvent (같은 correlation_id)

    Args:
        data: 단일 또는 멀티 심볼 데이터
        heartbeat_interval: 헬스체크 발행 주기 (bar 단위, 0=비활성)
    """

    def __init__(
        self,
        data: MarketDataSet | MultiSymbolData,
        heartbeat_interval: int = 0,
    ) -> None:
        self._data = data
        self._heartbeat_interval = heartbeat_interval
        self._bars_emitted: int = 0
        self._bus: EventBus | None = None

    async def start(self, bus: EventBus) -> None:
        """데이터 리플레이를 시작합니다.

        모든 bar를 순차적으로 BarEvent로 변환하여 EventBus에 발행합니다.

        Args:
            bus: 이벤트를 발행할 EventBus
        """
        self._bus = bus

        if self._is_multi_symbol:
            await self._replay_multi()
        else:
            await self._replay_single()

        logger.info(
            "HistoricalDataFeed finished: {} bars emitted",
            self._bars_emitted,
        )

    async def stop(self) -> None:
        """데이터 피드 중지."""
        self._bus = None

    @property
    def bars_emitted(self) -> int:
        """발행된 총 BarEvent 수."""
        return self._bars_emitted

    @property
    def _is_multi_symbol(self) -> bool:
        """멀티 심볼 데이터 여부."""
        from src.data.market_data import MultiSymbolData

        return isinstance(self._data, MultiSymbolData)

    async def _replay_single(self) -> None:
        """단일 심볼 리플레이."""
        from src.data.market_data import MarketDataSet

        assert isinstance(self._data, MarketDataSet)
        bus = self._bus
        assert bus is not None

        symbol = self._data.symbol
        timeframe = self._data.timeframe
        df = self._data.ohlcv

        for ts, row in df.iterrows():
            bar = BarEvent(
                symbol=symbol,
                timeframe=timeframe,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                bar_timestamp=ts.to_pydatetime(),  # type: ignore[union-attr]
                correlation_id=uuid4(),
                source="HistoricalDataFeed",
            )
            await bus.publish(bar)
            self._bars_emitted += 1

            # 각 bar의 이벤트 체인(Signal→Order→Fill) 완료 대기
            await bus.flush()

            await self._maybe_heartbeat(bus)

    async def _replay_multi(self) -> None:
        """멀티 심볼 리플레이.

        공통 인덱스를 기준으로 동일 타임스탬프의 모든 심볼에 대해
        같은 correlation_id를 가진 BarEvent를 발행합니다.
        """
        from src.data.market_data import MultiSymbolData

        assert isinstance(self._data, MultiSymbolData)
        bus = self._bus
        assert bus is not None

        symbols = self._data.symbols
        timeframe = self._data.timeframe
        ohlcv_dict = self._data.ohlcv

        # 공통 인덱스 사용 (첫 심볼 기준)
        common_index = ohlcv_dict[symbols[0]].index

        for ts in common_index:
            cid = uuid4()
            for symbol in symbols:
                df = ohlcv_dict[symbol]
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                bar = BarEvent(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    bar_timestamp=ts.to_pydatetime(),  # type: ignore[union-attr]
                    correlation_id=cid,
                    source="HistoricalDataFeed",
                )
                await bus.publish(bar)
                self._bars_emitted += 1

            # 타임스탬프 단위로 모든 심볼 이벤트 체인 완료 대기
            await bus.flush()

            await self._maybe_heartbeat(bus)

    async def _maybe_heartbeat(self, bus: EventBus) -> None:
        """주기적으로 HeartbeatEvent 발행."""
        if (
            self._heartbeat_interval > 0
            and self._bars_emitted % self._heartbeat_interval == 0
        ):
            hb = HeartbeatEvent(
                event_type=EventType.HEARTBEAT,
                component="HistoricalDataFeed",
                bars_processed=self._bars_emitted,
                source="HistoricalDataFeed",
            )
            await bus.publish(hb)
