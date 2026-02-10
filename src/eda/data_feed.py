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

import pandas as pd  # noqa: TC002 — runtime use in _resample_1m_to_tf
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


def _resample_1m_to_tf(df_1m: pd.DataFrame, freq: str) -> pd.DataFrame:
    """1m OHLCV DataFrame을 target TF로 pandas resample.

    Args:
        df_1m: 1m OHLCV DataFrame (DatetimeIndex)
        freq: pandas resample frequency ("1D", "4h", etc.)

    Returns:
        Resampled OHLCV DataFrame (NaN 행 제거)
    """
    resampled: pd.DataFrame = df_1m.resample(freq).agg(  # type: ignore[assignment]
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return resampled.dropna(subset=["close"])


class HistoricalDataFeed:
    """1m 히스토리컬 데이터를 CandleAggregator로 집계하여 재생.

    라이브 환경과 동일한 데이터 흐름을 백테스트에서 재현합니다:
    - 매 1m bar: BarEvent(tf="1m") 발행 -> PM intrabar SL/TS
    - TF candle 완성 시: BarEvent(tf=target_tf) 발행 -> Strategy + PM(full)
    - flush()는 TF candle 완성 시에만 호출 (성능 최적화)

    fast_mode=True일 때는 1m bar를 pandas resample로 사전 집계하여
    TF bar만 리플레이합니다. CandleAggregator를 바이패스하여
    이벤트 수를 ~1,189x 감소시킵니다 (1D 기준).

    Args:
        data: 1m 원본 데이터 (MarketDataSet 또는 MultiSymbolData)
        target_timeframe: 집계 목표 TF ("1D", "4h", "1h" 등)
        fast_mode: True면 pre-aggregation 후 TF bar만 리플레이 (intrabar SL/TS 없음)
    """

    def __init__(
        self,
        data: MarketDataSet | MultiSymbolData,
        target_timeframe: str,
        *,
        fast_mode: bool = False,
    ) -> None:
        self._data = data
        self._target_tf = target_timeframe
        self._fast_mode = fast_mode
        self._bars_emitted: int = 0

        if not fast_mode:
            from src.eda.candle_aggregator import CandleAggregator

            self._aggregator: CandleAggregator | None = CandleAggregator(target_timeframe)
        else:
            self._aggregator = None

    async def start(self, bus: EventBus) -> None:
        """1m bar 재생 + CandleAggregator 집계."""
        if self._fast_mode:
            if self._is_multi_symbol:
                await self._replay_multi_fast(bus)
            else:
                await self._replay_single_fast(bus)
        elif self._is_multi_symbol:
            await self._replay_multi(bus)
        else:
            await self._replay_single(bus)

        # 마지막 미완성 캔들 flush (fast_mode에서는 불필요)
        if self._aggregator is not None:
            remaining = self._aggregator.flush_all()
            for completed in remaining:
                await bus.publish(completed)
                self._bars_emitted += 1
                await bus.flush()

        mode_label = " [fast]" if self._fast_mode else ""
        logger.info(
            "HistoricalDataFeed finished{}: {} bars emitted (target_tf={})",
            mode_label,
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

    @property
    def _resample_freq(self) -> str:
        """target_timeframe → pandas resample frequency."""
        from src.eda.analytics import tf_to_pandas_freq

        return tf_to_pandas_freq(self._target_tf)

    # ------------------------------------------------------------------
    # Fast mode: pre-aggregated TF bars only
    # ------------------------------------------------------------------

    async def _replay_single_fast(self, bus: EventBus) -> None:
        """단일 심볼 fast mode: pandas resample 후 TF bar만 리플레이."""
        from src.data.market_data import MarketDataSet

        assert isinstance(self._data, MarketDataSet)

        symbol = self._data.symbol
        df_tf = _resample_1m_to_tf(self._data.ohlcv, self._resample_freq)

        for ts, row in df_tf.iterrows():
            o, h, lo, c, v = (
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            )
            if not validate_bar(o, h, lo, c, v, symbol, ts):
                continue

            bar = BarEvent(
                symbol=symbol,
                timeframe=self._target_tf,
                open=o,
                high=h,
                low=lo,
                close=c,
                volume=v,
                bar_timestamp=ts.to_pydatetime(),  # type: ignore[union-attr]
                correlation_id=uuid4(),
                source="HistoricalDataFeed",
            )
            await bus.publish(bar)
            self._bars_emitted += 1
            await bus.flush()

    async def _replay_multi_fast(self, bus: EventBus) -> None:
        """멀티 심볼 fast mode: pandas resample 후 common index TF bar 리플레이."""
        from src.data.market_data import MultiSymbolData

        assert isinstance(self._data, MultiSymbolData)

        symbols = self._data.symbols
        ohlcv_dict = self._data.ohlcv
        freq = self._resample_freq

        # 각 심볼을 TF로 resample
        resampled: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            resampled[sym] = _resample_1m_to_tf(ohlcv_dict[sym], freq)

        # common index 계산
        common_index = resampled[symbols[0]].index
        for sym in symbols[1:]:
            common_index = common_index.intersection(resampled[sym].index)

        for ts in common_index:
            cid = uuid4()

            for symbol in symbols:
                row = resampled[symbol].loc[ts]
                o, h, lo, c, v = (
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                )
                if not validate_bar(o, h, lo, c, v, symbol, ts):
                    continue

                bar = BarEvent(
                    symbol=symbol,
                    timeframe=self._target_tf,
                    open=o,
                    high=h,
                    low=lo,
                    close=c,
                    volume=v,
                    bar_timestamp=ts.to_pydatetime(),  # type: ignore[union-attr]
                    correlation_id=cid,
                    source="HistoricalDataFeed",
                )
                await bus.publish(bar)
                self._bars_emitted += 1

            await bus.flush()

    # ------------------------------------------------------------------
    # Standard mode: 1m replay + CandleAggregator
    # ------------------------------------------------------------------

    async def _replay_single(self, bus: EventBus) -> None:
        """단일 심볼 1m 리플레이 + 집계."""
        from src.data.market_data import MarketDataSet

        assert isinstance(self._data, MarketDataSet)
        assert self._aggregator is not None

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
        assert self._aggregator is not None

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
