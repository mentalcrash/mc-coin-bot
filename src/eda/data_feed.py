"""데이터 피드 — 1m 히스토리컬 데이터를 CandleAggregator로 집계하여 BarEvent로 변환.

HistoricalDataFeed: 1m 데이터를 CandleAggregator로 집계하여 TF bar를 생성합니다.
멀티 심볼 데이터의 경우 동일 타임스탬프에 N개 BarEvent를 발행합니다.

Rules Applied:
    - #12 Data Engineering: UTC, DatetimeIndex
    - #10 Python Standards: Protocol for interface
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from src.core.events import BarEvent

if TYPE_CHECKING:
    from src.core.event_bus import EventBus
    from src.data.market_data import MarketDataSet, MultiSymbolData


def validate_bar(
    o: float, h: float, lo: float, c: float, v: float, symbol: str, ts: object
) -> bool:
    """OHLCV bar 데이터 품질 검증 (M-004).

    Returns:
        True if valid, False if bar should be skipped.
    """
    for name, val in [("open", o), ("high", h), ("low", lo), ("close", c), ("volume", v)]:
        if math.isnan(val) or math.isinf(val):
            logger.warning("Invalid {} ({}) for {} at {}, skipping bar", name, val, symbol, ts)
            return False
    if h < lo:
        logger.warning("high ({}) < low ({}) for {} at {}, skipping bar", h, lo, symbol, ts)
        return False
    return True


class HistoricalDataFeed:
    """1m 히스토리컬 데이터를 CandleAggregator로 집계하여 재생.

    라이브 환경과 동일한 데이터 흐름을 백테스트에서 재현합니다:
    - 매 1m bar: BarEvent(tf="1m") 발행 → PM intrabar SL/TS
    - TF candle 완성 시: BarEvent(tf=target_tf) 발행 → Strategy + PM(full)
    - flush()는 TF candle 완성 시에만 호출 (성능 최적화)

    Args:
        data: 1m 원본 데이터 (MarketDataSet 또는 MultiSymbolData)
        target_timeframe: 집계 목표 TF ("1D", "4h", "1h" 등)
    """

    def __init__(
        self,
        data: MarketDataSet | MultiSymbolData,
        target_timeframe: str,
    ) -> None:
        from src.eda.candle_aggregator import CandleAggregator

        self._data = data
        self._target_tf = target_timeframe
        self._aggregator = CandleAggregator(target_timeframe)
        self._bars_emitted: int = 0

    async def start(self, bus: EventBus) -> None:
        """1m bar 재생 + CandleAggregator 집계."""
        if self._is_multi_symbol:
            await self._replay_multi(bus)
        else:
            await self._replay_single(bus)

        # 마지막 미완성 캔들 flush
        remaining = self._aggregator.flush_all()
        for completed in remaining:
            await bus.publish(completed)
            self._bars_emitted += 1
            await bus.flush()

        logger.info(
            "HistoricalDataFeed finished: {} bars emitted (target_tf={})",
            self._bars_emitted,
            self._target_tf,
        )

    async def stop(self) -> None:
        """데이터 피드 중지."""

    @property
    def bars_emitted(self) -> int:
        """발행된 총 BarEvent 수."""
        return self._bars_emitted

    @property
    def _is_multi_symbol(self) -> bool:
        """멀티 심볼 데이터 여부."""
        from src.data.market_data import MultiSymbolData

        return isinstance(self._data, MultiSymbolData)

    async def _replay_single(self, bus: EventBus) -> None:
        """단일 심볼 1m 리플레이 + 집계."""
        from src.data.market_data import MarketDataSet

        assert isinstance(self._data, MarketDataSet)

        symbol = self._data.symbol
        df = self._data.ohlcv

        for ts, row in df.iterrows():
            o, h, lo, c, v = (
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            )
            if not validate_bar(o, h, lo, c, v, symbol, ts):
                continue

            # 1. 1m BarEvent 발행
            bar_1m = BarEvent(
                symbol=symbol,
                timeframe="1m",
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=v,
                bar_timestamp=ts.to_pydatetime(),  # type: ignore[union-attr]
                correlation_id=uuid4(),
                source="HistoricalDataFeed",
            )
            await bus.publish(bar_1m)
            self._bars_emitted += 1

            # 2. CandleAggregator에 전달
            completed = self._aggregator.on_1m_bar(bar_1m)
            if completed is not None:
                # 3. 완성된 TF BarEvent 발행
                await bus.publish(completed)
                self._bars_emitted += 1
                # 4. Signal→Order→Fill 체인 완료 대기
                await bus.flush()

    async def _replay_multi(self, bus: EventBus) -> None:
        """멀티 심볼 1m 리플레이 + 집계.

        동일 타임스탬프의 모든 심볼 1m bar를 같은 correlation_id로 발행합니다.
        """
        from src.data.market_data import MultiSymbolData

        assert isinstance(self._data, MultiSymbolData)

        symbols = self._data.symbols
        ohlcv_dict = self._data.ohlcv

        common_index = ohlcv_dict[symbols[0]].index

        for ts in common_index:
            cid = uuid4()
            any_completed = False

            for symbol in symbols:
                df = ohlcv_dict[symbol]
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                o, h, lo, c, v = (
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                )
                if not validate_bar(o, h, lo, c, v, symbol, ts):
                    continue

                # 1. 1m BarEvent 발행
                bar_1m = BarEvent(
                    symbol=symbol,
                    timeframe="1m",
                    open=o,
                    high=h,
                    low=lo,
                    close=c,
                    volume=v,
                    bar_timestamp=ts.to_pydatetime(),  # type: ignore[union-attr]
                    correlation_id=cid,
                    source="HistoricalDataFeed",
                )
                await bus.publish(bar_1m)
                self._bars_emitted += 1

                # 2. CandleAggregator에 전달
                completed = self._aggregator.on_1m_bar(bar_1m)
                if completed is not None:
                    await bus.publish(completed)
                    self._bars_emitted += 1
                    any_completed = True

            # TF candle이 완성된 경우에만 flush
            if any_completed:
                await bus.flush()
