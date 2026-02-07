"""CandleAggregator 단위 테스트.

1m BarEvent → target TF candle 집계 로직을 검증합니다.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from src.core.events import BarEvent
from src.eda.candle_aggregator import CandleAggregator, _timeframe_to_seconds


# =========================================================================
# Helpers
# =========================================================================
def _make_1m_bar(
    symbol: str,
    ts: datetime,
    open_: float = 50000.0,
    high: float = 50100.0,
    low: float = 49900.0,
    close: float = 50050.0,
    volume: float = 10.0,
) -> BarEvent:
    """1m BarEvent 생성."""
    return BarEvent(
        symbol=symbol,
        timeframe="1m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bar_timestamp=ts,
        correlation_id=uuid4(),
        source="test",
    )


# =========================================================================
# _timeframe_to_seconds
# =========================================================================
class TestTimeframeToSeconds:
    """_timeframe_to_seconds 유틸리티 함수 테스트."""

    def test_1m(self) -> None:
        assert _timeframe_to_seconds("1m") == 60

    def test_5m(self) -> None:
        assert _timeframe_to_seconds("5m") == 300

    def test_15m(self) -> None:
        assert _timeframe_to_seconds("15m") == 900

    def test_1h(self) -> None:
        assert _timeframe_to_seconds("1h") == 3600

    def test_4h(self) -> None:
        assert _timeframe_to_seconds("4h") == 14400

    def test_1d_uppercase(self) -> None:
        assert _timeframe_to_seconds("1D") == 86400

    def test_1d_lowercase(self) -> None:
        assert _timeframe_to_seconds("1d") == 86400

    def test_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            _timeframe_to_seconds("3h")


# =========================================================================
# 1h 집계 테스트
# =========================================================================
class TestAggregator1h:
    """1m → 1h 집계 테스트."""

    def test_60_bars_to_1_candle(self) -> None:
        """60개 1m bar → 1개 1h candle."""
        agg = CandleAggregator("1h")
        base = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)  # 12:00 UTC

        results: list[BarEvent] = []
        for i in range(60):
            ts = base + timedelta(minutes=i)
            bar = _make_1m_bar("BTC/USDT", ts, volume=1.0)
            completed = agg.on_1m_bar(bar)
            if completed is not None:
                results.append(completed)

        # 60개 bar가 모두 같은 기간(12:00-13:00)이므로 아직 미완성
        assert len(results) == 0

        # 다음 기간 bar 하나 → 이전 캔들 완성
        next_ts = base + timedelta(minutes=60)
        completed = agg.on_1m_bar(_make_1m_bar("BTC/USDT", next_ts))
        assert completed is not None
        assert completed.timeframe == "1h"
        results.append(completed)

        assert len(results) == 1
        candle = results[0]
        assert candle.volume == 60.0  # 60 * 1.0
        assert candle.bar_timestamp == datetime(2024, 6, 15, 13, 0, tzinfo=UTC)

    def test_ohlcv_correctness(self) -> None:
        """OHLCV 정합성: O=first, H=max, L=min, C=last, V=sum."""
        agg = CandleAggregator("1h")
        base = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

        prices = [
            (100.0, 105.0, 98.0, 102.0, 5.0),   # bar 0: O=100, H=105, L=98, C=102, V=5
            (102.0, 110.0, 101.0, 108.0, 3.0),  # bar 1: H=110 → new high
            (108.0, 109.0, 95.0, 96.0, 7.0),    # bar 2: L=95 → new low, C=96 → last close
        ]

        for i, (o, h, lo, c, v) in enumerate(prices):
            ts = base + timedelta(minutes=i)
            agg.on_1m_bar(_make_1m_bar("BTC/USDT", ts, open_=o, high=h, low=lo, close=c, volume=v))

        # flush로 미완성 캔들 완성
        candle = agg.flush_partial("BTC/USDT")
        assert candle is not None
        assert candle.open == 100.0   # first open
        assert candle.high == 110.0   # max high
        assert candle.low == 95.0     # min low
        assert candle.close == 96.0   # last close
        assert candle.volume == 15.0  # sum volume


# =========================================================================
# 4h 집계 테스트
# =========================================================================
class TestAggregator4h:
    """1m → 4h 집계 테스트."""

    def test_240_bars_to_1_candle(self) -> None:
        """240개 1m bar → 1개 4h candle."""
        agg = CandleAggregator("4h")
        base = datetime(2024, 6, 15, 8, 0, tzinfo=UTC)  # 08:00 UTC

        results: list[BarEvent] = []
        for i in range(240):
            ts = base + timedelta(minutes=i)
            bar = _make_1m_bar("BTC/USDT", ts, volume=1.0)
            completed = agg.on_1m_bar(bar)
            if completed is not None:
                results.append(completed)

        assert len(results) == 0  # 아직 미완성 (다음 기간 진입 없음)

        # 다음 기간 bar → 완성
        next_ts = base + timedelta(minutes=240)
        completed = agg.on_1m_bar(_make_1m_bar("BTC/USDT", next_ts))
        assert completed is not None
        assert completed.timeframe == "4h"
        assert completed.volume == 240.0
        # period_end = 08:00 + 4h = 12:00
        assert completed.bar_timestamp == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)


# =========================================================================
# 1D 집계 테스트
# =========================================================================
class TestAggregator1D:
    """1m → 1D 집계 테스트."""

    def test_1440_bars_to_1_candle(self) -> None:
        """1440개 1m bar → 1개 1D candle."""
        agg = CandleAggregator("1D")
        base = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)  # UTC 00:00

        results: list[BarEvent] = []
        for i in range(1440):
            ts = base + timedelta(minutes=i)
            bar = _make_1m_bar("BTC/USDT", ts, volume=0.5)
            completed = agg.on_1m_bar(bar)
            if completed is not None:
                results.append(completed)

        assert len(results) == 0

        # 다음 날 첫 bar → 완성
        next_day = datetime(2024, 6, 16, 0, 0, tzinfo=UTC)
        completed = agg.on_1m_bar(_make_1m_bar("BTC/USDT", next_day))
        assert completed is not None
        assert completed.timeframe == "1D"
        assert completed.volume == pytest.approx(720.0)  # 1440 * 0.5
        assert completed.bar_timestamp == datetime(2024, 6, 16, 0, 0, tzinfo=UTC)

    def test_utc_midnight_boundary(self) -> None:
        """1D는 UTC 00:00 경계로 정렬."""
        agg = CandleAggregator("1D")

        # 23:59 → 6/15 기간
        bar_2359 = _make_1m_bar(
            "BTC/USDT",
            datetime(2024, 6, 15, 23, 59, tzinfo=UTC),
        )
        assert agg.on_1m_bar(bar_2359) is None

        # 00:00 → 6/16 기간 → 이전 캔들 완성
        bar_0000 = _make_1m_bar(
            "BTC/USDT",
            datetime(2024, 6, 16, 0, 0, tzinfo=UTC),
        )
        completed = agg.on_1m_bar(bar_0000)
        assert completed is not None
        assert completed.bar_timestamp == datetime(2024, 6, 16, 0, 0, tzinfo=UTC)


# =========================================================================
# 기간 전환 테스트
# =========================================================================
class TestPeriodTransition:
    """기간 전환 시 이전 candle 완성 + 새 candle 시작."""

    def test_transition_returns_completed_candle(self) -> None:
        """새 기간 진입 시 이전 캔들을 BarEvent로 반환."""
        agg = CandleAggregator("1h")
        base = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

        # 12:00~12:59 범위 3개 bar
        for i in range(3):
            agg.on_1m_bar(_make_1m_bar("BTC/USDT", base + timedelta(minutes=i)))

        # 13:00 → 기간 전환
        completed = agg.on_1m_bar(
            _make_1m_bar("BTC/USDT", datetime(2024, 6, 15, 13, 0, tzinfo=UTC))
        )
        assert completed is not None
        assert completed.timeframe == "1h"

    def test_none_within_same_period(self) -> None:
        """같은 기간 내에서는 None 반환."""
        agg = CandleAggregator("1h")
        base = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

        for i in range(30):
            result = agg.on_1m_bar(_make_1m_bar("BTC/USDT", base + timedelta(minutes=i)))
            assert result is None


# =========================================================================
# Gap 처리 테스트
# =========================================================================
class TestGapHandling:
    """1m bar 누락(gap) 처리."""

    def test_gap_completes_previous_candle(self) -> None:
        """큰 gap이 있어도 기간 전환 시 이전 캔들 완성."""
        agg = CandleAggregator("1h")

        # 12:00에 bar 1개
        agg.on_1m_bar(_make_1m_bar(
            "BTC/USDT",
            datetime(2024, 6, 15, 12, 0, tzinfo=UTC),
            open_=100.0, high=105.0, low=99.0, close=103.0, volume=10.0,
        ))

        # 3시간 gap → 15:00에 다음 bar
        completed = agg.on_1m_bar(_make_1m_bar(
            "BTC/USDT",
            datetime(2024, 6, 15, 15, 0, tzinfo=UTC),
            open_=200.0, high=210.0, low=195.0, close=205.0, volume=20.0,
        ))

        assert completed is not None
        assert completed.open == 100.0  # 12:00 기간의 유일한 bar
        assert completed.close == 103.0
        assert completed.volume == 10.0  # 1개 bar만


# =========================================================================
# 멀티 심볼 독립 집계 테스트
# =========================================================================
class TestMultiSymbol:
    """심볼별 독립 집계."""

    def test_independent_aggregation(self) -> None:
        """서로 다른 심볼은 독립적으로 집계."""
        agg = CandleAggregator("1h")
        base = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

        # BTC: 30분 데이터
        for i in range(30):
            agg.on_1m_bar(_make_1m_bar("BTC/USDT", base + timedelta(minutes=i), volume=1.0))

        # ETH: 60분 데이터
        for i in range(60):
            agg.on_1m_bar(_make_1m_bar("ETH/USDT", base + timedelta(minutes=i), volume=2.0))

        # BTC flush
        btc_candle = agg.flush_partial("BTC/USDT")
        assert btc_candle is not None
        assert btc_candle.symbol == "BTC/USDT"
        assert btc_candle.volume == 30.0

        # ETH flush
        eth_candle = agg.flush_partial("ETH/USDT")
        assert eth_candle is not None
        assert eth_candle.symbol == "ETH/USDT"
        assert eth_candle.volume == 120.0  # 60 * 2.0

    def test_flush_all(self) -> None:
        """flush_all: 모든 심볼 동시 완성."""
        agg = CandleAggregator("1h")
        base = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

        agg.on_1m_bar(_make_1m_bar("BTC/USDT", base))
        agg.on_1m_bar(_make_1m_bar("ETH/USDT", base))

        results = agg.flush_all()
        assert len(results) == 2
        symbols = {r.symbol for r in results}
        assert symbols == {"BTC/USDT", "ETH/USDT"}


# =========================================================================
# flush_partial 테스트
# =========================================================================
class TestFlushPartial:
    """flush_partial 동작 검증."""

    def test_flush_nonexistent_symbol(self) -> None:
        """존재하지 않는 심볼 flush → None."""
        agg = CandleAggregator("1h")
        assert agg.flush_partial("NONEXIST/USDT") is None

    def test_flush_clears_partial(self) -> None:
        """flush 후 partial 제거."""
        agg = CandleAggregator("1h")
        base = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        agg.on_1m_bar(_make_1m_bar("BTC/USDT", base))

        first = agg.flush_partial("BTC/USDT")
        assert first is not None
        second = agg.flush_partial("BTC/USDT")
        assert second is None


# =========================================================================
# _align_to_period 테스트
# =========================================================================
class TestAlignToPeriod:
    """UTC 경계 정렬 테스트."""

    def test_1h_alignment(self) -> None:
        """1h: 12:35 → 12:00."""
        agg = CandleAggregator("1h")
        ts = datetime(2024, 6, 15, 12, 35, tzinfo=UTC)
        aligned = agg._align_to_period(ts)
        assert aligned == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

    def test_4h_alignment(self) -> None:
        """4h: 14:30 → 12:00."""
        agg = CandleAggregator("4h")
        ts = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        aligned = agg._align_to_period(ts)
        assert aligned == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

    def test_1d_alignment(self) -> None:
        """1D: 15:30 → 00:00."""
        agg = CandleAggregator("1D")
        ts = datetime(2024, 6, 15, 15, 30, tzinfo=UTC)
        aligned = agg._align_to_period(ts)
        assert aligned == datetime(2024, 6, 15, 0, 0, tzinfo=UTC)

    def test_exact_boundary(self) -> None:
        """정확히 경계에 있는 경우."""
        agg = CandleAggregator("1h")
        ts = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)
        aligned = agg._align_to_period(ts)
        assert aligned == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)


# =========================================================================
# 연속 기간 테스트
# =========================================================================
class TestMultiplePeriods:
    """여러 기간에 걸친 연속 집계."""

    def test_three_consecutive_hours(self) -> None:
        """3시간 연속 데이터 → 2개 완성 + 1개 미완성."""
        agg = CandleAggregator("1h")
        base = datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

        completed: list[BarEvent] = []
        # 180분 = 3시간
        for i in range(180):
            ts = base + timedelta(minutes=i)
            result = agg.on_1m_bar(_make_1m_bar("BTC/USDT", ts, volume=1.0))
            if result is not None:
                completed.append(result)

        # 12:00-13:00, 13:00-14:00 → 2개 완성 (세 번째 기간 진입 시 두 번째 완성)
        assert len(completed) == 2
        assert completed[0].bar_timestamp == datetime(2024, 6, 15, 13, 0, tzinfo=UTC)
        assert completed[1].bar_timestamp == datetime(2024, 6, 15, 14, 0, tzinfo=UTC)

        # 미완성 1개 flush
        last = agg.flush_partial("BTC/USDT")
        assert last is not None
        assert last.bar_timestamp == datetime(2024, 6, 15, 15, 0, tzinfo=UTC)

    def test_two_consecutive_days(self) -> None:
        """2일 연속 1440분씩 → 1개 완성 + 1개 미완성."""
        agg = CandleAggregator("1D")
        day1 = datetime(2024, 6, 15, 0, 0, tzinfo=UTC)

        completed: list[BarEvent] = []
        # 2일 = 2880분
        for i in range(2880):
            ts = day1 + timedelta(minutes=i)
            result = agg.on_1m_bar(_make_1m_bar("BTC/USDT", ts, volume=0.1))
            if result is not None:
                completed.append(result)

        assert len(completed) == 1
        assert completed[0].bar_timestamp == datetime(2024, 6, 16, 0, 0, tzinfo=UTC)
        assert completed[0].volume == pytest.approx(144.0)  # 1440 * 0.1

        last = agg.flush_partial("BTC/USDT")
        assert last is not None


# =========================================================================
# Properties
# =========================================================================
class TestProperties:
    """CandleAggregator 속성 테스트."""

    def test_target_timeframe(self) -> None:
        agg = CandleAggregator("4h")
        assert agg.target_timeframe == "4h"
