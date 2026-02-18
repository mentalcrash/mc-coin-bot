"""MultiTimeframeCandleAggregator 단위 테스트.

여러 TF의 CandleAggregator를 합성하여 1m bar에서 복수 TF bar를 생성하는 로직을 검증합니다.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from src.core.events import BarEvent
from src.eda.candle_aggregator import MultiTimeframeCandleAggregator

# ── Helpers ──────────────────────────────────────────────────────


def _make_1m_bar(
    symbol: str,
    ts: datetime,
    open_: float = 50000.0,
    high: float = 50100.0,
    low: float = 49900.0,
    close: float = 50050.0,
    volume: float = 10.0,
) -> BarEvent:
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


# ── Tests ────────────────────────────────────────────────────────


class TestMultiTimeframeCandleAggregator:
    """MultiTimeframeCandleAggregator 테스트."""

    def test_init_sorts_by_tf_seconds(self) -> None:
        """TF가 초 오름차순으로 정렬되어야 함."""
        agg = MultiTimeframeCandleAggregator({"1D", "4h"})
        assert agg.timeframes == ["4h", "1D"]

    def test_single_tf_same_as_candle_aggregator(self) -> None:
        """단일 TF → 기존 CandleAggregator와 동일 동작."""
        agg = MultiTimeframeCandleAggregator({"1D"})
        assert agg.timeframes == ["1D"]

        # 1D = 1440분. 새 기간 진입 시에만 이전 캔들 완성.
        base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        # 첫 1440분 → 아직 완성 안됨
        for i in range(1440):
            result = agg.on_1m_bar(_make_1m_bar("BTC/USDT", base + timedelta(minutes=i)))
            assert result == []

        # 다음 날 첫 bar → 이전 1D 캔들 완성
        result = agg.on_1m_bar(_make_1m_bar("BTC/USDT", base + timedelta(days=1)))
        assert len(result) == 1
        assert result[0].timeframe == "1D"

    def test_4h_1d_dual_tf(self) -> None:
        """4h + 1D: 4h는 6x/day, 1D는 1x/day 완성."""
        agg = MultiTimeframeCandleAggregator({"4h", "1D"})

        base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        completed_4h: list[BarEvent] = []
        completed_1d: list[BarEvent] = []

        # 2일치 1m bars 공급 (2 * 1440 = 2880 bars + 1 extra)
        for i in range(2880 + 1):
            results = agg.on_1m_bar(_make_1m_bar("BTC/USDT", base + timedelta(minutes=i)))
            for bar in results:
                if bar.timeframe == "4h":
                    completed_4h.append(bar)
                elif bar.timeframe == "1D":
                    completed_1d.append(bar)

        # 2일: 4h bar = 12 (6/day * 2), 1D bar = 2
        assert len(completed_4h) == 12
        assert len(completed_1d) == 2

    def test_simultaneous_completion_order(self) -> None:
        """UTC 00:00에서 4h + 1D 동시 완성 시, 4h가 먼저 emit."""
        agg = MultiTimeframeCandleAggregator({"4h", "1D"})

        base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        # 1일치 1m bars 공급
        for i in range(1440):
            agg.on_1m_bar(_make_1m_bar("BTC/USDT", base + timedelta(minutes=i)))

        # 다음 날 첫 bar → 4h + 1D 동시 완성
        results = agg.on_1m_bar(_make_1m_bar("BTC/USDT", base + timedelta(days=1)))
        assert len(results) == 2
        assert results[0].timeframe == "4h"
        assert results[1].timeframe == "1D"

    def test_flush_all_returns_all_tf_partials(self) -> None:
        """flush_all()이 모든 TF의 미완성 캔들을 반환."""
        agg = MultiTimeframeCandleAggregator({"4h", "1D"})

        base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        # 일부 bars 공급 (완성 안 됨)
        for i in range(10):
            agg.on_1m_bar(_make_1m_bar("BTC/USDT", base + timedelta(minutes=i)))

        results = agg.flush_all()
        # 4h + 1D 각 1개씩 미완성
        assert len(results) == 2
        tfs = {bar.timeframe for bar in results}
        assert tfs == {"4h", "1D"}

    def test_multi_symbol(self) -> None:
        """여러 심볼에 대해 각각 독립적으로 집계."""
        agg = MultiTimeframeCandleAggregator({"4h"})

        base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        btc_completed = 0
        eth_completed = 0

        # 4h = 240분. 240분 공급 후 새 기간 진입 시 완성.
        for i in range(241):
            ts = base + timedelta(minutes=i)
            for _bar in agg.on_1m_bar(_make_1m_bar("BTC/USDT", ts)):
                btc_completed += 1
            for _bar in agg.on_1m_bar(_make_1m_bar("ETH/USDT", ts)):
                eth_completed += 1

        assert btc_completed == 1
        assert eth_completed == 1

    def test_empty_on_1m_bar(self) -> None:
        """첫 번째 bar에서는 빈 리스트 반환."""
        agg = MultiTimeframeCandleAggregator({"4h", "1D"})
        base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        results = agg.on_1m_bar(_make_1m_bar("BTC/USDT", base))
        assert results == []


class TestMultiTimeframeEdgeCases:
    """엣지 케이스 테스트."""

    def test_three_timeframes(self) -> None:
        """1h + 4h + 1D 3종 TF 정렬."""
        agg = MultiTimeframeCandleAggregator({"1D", "1h", "4h"})
        assert agg.timeframes == ["1h", "4h", "1D"]

    def test_unsupported_tf_raises(self) -> None:
        """지원하지 않는 TF가 포함되면 ValueError."""
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            MultiTimeframeCandleAggregator({"1D", "3D"})
