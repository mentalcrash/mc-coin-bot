"""AnalyticsEngine 메모리 제한 테스트.

deque(maxlen=N)을 사용한 eviction이 올바르게 동작하는지 검증합니다.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.core.events import BalanceUpdateEvent, BarEvent, FillEvent
from src.eda.analytics import AnalyticsEngine, EquityPoint
from src.models.backtest import TradeRecord


class TestAnalyticsMemoryLimits:
    """AnalyticsEngine deque maxlen eviction 검증."""

    def test_equity_curve_bounded(self) -> None:
        """max_equity_points 초과 시 오래된 포인트가 제거된다."""
        engine = AnalyticsEngine(10_000.0, max_equity_points=5)

        base = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(10):
            ts = base + timedelta(hours=i)
            engine._equity_curve.append(EquityPoint(timestamp=ts, equity=10_000.0 + i))

        assert len(engine.equity_curve) == 5
        # 가장 오래된 것이 제거되고, 최신 5개만 남음
        assert engine.equity_curve[0].equity == 10_005.0
        assert engine.equity_curve[-1].equity == 10_009.0

    def test_closed_trades_bounded(self) -> None:
        """max_closed_trades 초과 시 오래된 trade가 제거된다."""
        engine = AnalyticsEngine(10_000.0, max_closed_trades=3)

        base = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(7):
            trade = TradeRecord(
                entry_time=base + timedelta(hours=i),
                exit_time=base + timedelta(hours=i, minutes=30),
                symbol="BTC/USDT",
                direction="LONG",
                entry_price=Decimal(50000),
                exit_price=Decimal(51000),
                size=Decimal("0.1"),
                pnl=Decimal(str(100 * i)),
                pnl_pct=0.02,
                fees=Decimal(5),
            )
            engine._closed_trades.append(trade)

        assert len(engine.closed_trades) == 3
        assert engine.closed_trades[0].pnl == Decimal(400)
        assert engine.closed_trades[-1].pnl == Decimal(600)

    def test_bar_timestamps_bounded(self) -> None:
        """max_bar_timestamps 초과 시 오래된 timestamp가 제거된다."""
        engine = AnalyticsEngine(10_000.0, max_bar_timestamps=4)

        base = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(8):
            engine._bar_timestamps.append(base + timedelta(days=i))

        assert len(engine._bar_timestamps) == 4
        assert engine._bar_timestamps[0] == base + timedelta(days=4)

    @pytest.mark.asyncio
    async def test_equity_eviction_via_event(self) -> None:
        """BalanceUpdateEvent를 통한 equity 추가도 maxlen으로 제한된다."""
        engine = AnalyticsEngine(10_000.0, max_equity_points=3)

        base = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(5):
            event = BalanceUpdateEvent(
                timestamp=base + timedelta(hours=i),
                total_equity=10_000.0 + i * 100,
                available_cash=5_000.0,
                source="test",
            )
            await engine._on_balance_update(event)

        assert len(engine.equity_curve) == 3
        assert engine.equity_curve[0].equity == 10_200.0

    @pytest.mark.asyncio
    async def test_bar_timestamp_eviction_via_event(self) -> None:
        """BarEvent를 통한 timestamp 추가도 maxlen으로 제한된다."""
        engine = AnalyticsEngine(10_000.0, max_bar_timestamps=3)

        base = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(6):
            event = BarEvent(
                symbol="BTC/USDT",
                timeframe="1D",
                open=50000.0,
                high=51000.0,
                low=49000.0,
                close=50500.0,
                volume=100.0,
                bar_timestamp=base + timedelta(days=i),
                source="test",
            )
            await engine._on_bar(event)

        assert engine.bar_count == 3

    @pytest.mark.asyncio
    async def test_trade_eviction_via_fill(self) -> None:
        """FillEvent를 통한 trade close도 maxlen으로 제한된다."""
        engine = AnalyticsEngine(10_000.0, max_closed_trades=2)

        base = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(4):
            # Open: BUY
            buy = FillEvent(
                client_order_id=f"buy-{i}",
                symbol=f"SYM{i}/USDT",
                side="BUY",
                fill_price=100.0 + i,
                fill_qty=1.0,
                fee=0.1,
                fill_timestamp=base + timedelta(hours=i * 2),
                source="test",
            )
            await engine._on_fill(buy)

            # Close: SELL
            sell = FillEvent(
                client_order_id=f"sell-{i}",
                symbol=f"SYM{i}/USDT",
                side="SELL",
                fill_price=110.0 + i,
                fill_qty=1.0,
                fee=0.1,
                fill_timestamp=base + timedelta(hours=i * 2 + 1),
                source="test",
            )
            await engine._on_fill(sell)

        assert len(engine.closed_trades) == 2

    def test_default_maxlen_values(self) -> None:
        """기본 maxlen 값이 올바르게 설정된다."""
        engine = AnalyticsEngine(10_000.0)

        assert engine._equity_curve.maxlen == 10_000
        assert engine._closed_trades.maxlen == 50_000
        assert engine._bar_timestamps.maxlen == 10_000

    def test_deque_supports_iteration_and_indexing(self) -> None:
        """deque가 list와 동일하게 iteration과 인덱싱을 지원한다."""
        engine = AnalyticsEngine(10_000.0, max_equity_points=5)

        base = datetime(2025, 1, 1, tzinfo=UTC)
        for i in range(5):
            engine._equity_curve.append(
                EquityPoint(timestamp=base + timedelta(hours=i), equity=float(i))
            )

        # Iteration
        values = [p.equity for p in engine.equity_curve]
        assert values == [0.0, 1.0, 2.0, 3.0, 4.0]

        # Indexing
        assert engine.equity_curve[0].equity == 0.0
        assert engine.equity_curve[-1].equity == 4.0

        # len
        assert len(engine.equity_curve) == 5
