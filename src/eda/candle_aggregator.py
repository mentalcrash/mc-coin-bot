"""CandleAggregator — 1m BarEvent를 target TF candle로 집계.

순수 로직 컴포넌트 (I/O 없음, EventBus 의존 없음).
1m BarEvent를 수신하여 target timeframe(1h, 4h, 1D 등)의 완성된 BarEvent를 생성합니다.

Rules Applied:
    - Pure Logic: 외부 의존성 없음, 단위 테스트 용이
    - UTC Boundary: 모든 TF는 UTC 기준 정렬
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from src.core.events import BarEvent

# 지원하는 타임프레임 → 초 변환
_TF_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1D": 86400,
    "1d": 86400,
}


def _timeframe_to_seconds(tf: str) -> int:
    """타임프레임 문자열 → 초 단위.

    Args:
        tf: 타임프레임 문자열 ("1m", "5m", "15m", "30m", "1h", "4h", "1D")

    Returns:
        초 단위 정수

    Raises:
        ValueError: 지원하지 않는 타임프레임
    """
    if tf in _TF_SECONDS:
        return _TF_SECONDS[tf]
    msg = f"Unsupported timeframe: {tf}. Supported: {list(_TF_SECONDS.keys())}"
    raise ValueError(msg)


@dataclass
class PartialCandle:
    """집계 중인 미완성 캔들.

    Attributes:
        symbol: 거래 심볼
        timeframe: 타임프레임 문자열
        open: 시가
        high: 고가
        low: 저가
        close: 종가
        volume: 거래량
        period_start: TF 기간 시작 (UTC boundary)
        period_end: TF 기간 종료 (이 시각에 완성)
        bar_count: 집계된 1m bar 수
    """

    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    period_start: datetime
    period_end: datetime
    bar_count: int = 0


class CandleAggregator:
    """1m BarEvent → target TF candle 집계.

    각 심볼별로 독립적인 PartialCandle을 관리하며,
    TF 기간이 완성되면 BarEvent를 반환합니다.

    Args:
        target_timeframe: 집계 목표 타임프레임 ("1h", "4h", "1D" 등)
    """

    def __init__(self, target_timeframe: str) -> None:
        self._target_tf = target_timeframe
        self._tf_seconds = _timeframe_to_seconds(target_timeframe)
        self._partials: dict[str, PartialCandle] = {}

    @property
    def target_timeframe(self) -> str:
        """집계 목표 타임프레임."""
        return self._target_tf

    def on_1m_bar(self, bar: BarEvent) -> BarEvent | None:
        """1m bar 수신 → target TF candle에 집계.

        새 기간 진입 시 이전 캔들을 완성하여 BarEvent로 반환합니다.
        같은 기간 내에서는 OHLCV를 업데이트하고 None을 반환합니다.

        Args:
            bar: 1m BarEvent

        Returns:
            완성된 TF BarEvent (미완성 시 None)
        """
        symbol = bar.symbol
        period_start = self._align_to_period(bar.bar_timestamp)
        period_end = period_start + timedelta(seconds=self._tf_seconds)

        partial = self._partials.get(symbol)

        # 새 기간 진입 → 이전 캔들 완성
        completed: BarEvent | None = None
        if partial is not None and period_start != partial.period_start:
            completed = self._finalize(partial)
            partial = None

        # 새 캔들 시작 또는 기존 캔들에 집계
        if partial is None:
            self._partials[symbol] = PartialCandle(
                symbol=symbol,
                timeframe=self._target_tf,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                period_start=period_start,
                period_end=period_end,
                bar_count=1,
            )
        else:
            partial.high = max(partial.high, bar.high)
            partial.low = min(partial.low, bar.low)
            partial.close = bar.close
            partial.volume += bar.volume
            partial.bar_count += 1

        return completed

    def flush_partial(self, symbol: str) -> BarEvent | None:
        """미완성 캔들 강제 완성 (shutdown 시).

        Args:
            symbol: 심볼

        Returns:
            완성된 BarEvent (미완성 캔들이 없으면 None)
        """
        partial = self._partials.pop(symbol, None)
        if partial is None:
            return None
        return self._finalize(partial)

    def flush_all(self) -> list[BarEvent]:
        """모든 심볼의 미완성 캔들 강제 완성.

        Returns:
            완성된 BarEvent 리스트
        """
        results: list[BarEvent] = []
        for symbol in list(self._partials.keys()):
            bar = self.flush_partial(symbol)
            if bar is not None:
                results.append(bar)
        return results

    def _align_to_period(self, ts: datetime) -> datetime:
        """타임스탬프를 TF 기간 시작에 정렬 (UTC boundary).

        1D: UTC 00:00
        4h: 00/04/08/12/16/20
        1h: 매 정시
        etc.

        Args:
            ts: UTC 타임스탬프

        Returns:
            기간 시작 시각
        """
        # epoch 기반 정렬 (UTC 기준)
        epoch = datetime(2000, 1, 1, tzinfo=UTC)
        total_seconds = int((ts - epoch).total_seconds())
        aligned_seconds = (total_seconds // self._tf_seconds) * self._tf_seconds
        return epoch + timedelta(seconds=aligned_seconds)

    def _finalize(self, partial: PartialCandle) -> BarEvent:
        """PartialCandle → BarEvent 변환.

        Args:
            partial: 완성할 PartialCandle

        Returns:
            완성된 BarEvent
        """
        return BarEvent(
            symbol=partial.symbol,
            timeframe=partial.timeframe,
            open=partial.open,
            high=partial.high,
            low=partial.low,
            close=partial.close,
            volume=partial.volume,
            bar_timestamp=partial.period_end,
            correlation_id=uuid4(),
            source="CandleAggregator",
        )
