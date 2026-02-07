"""AnalyticsEngine 테스트.

Equity curve 기록, trade 기록 생성, PerformanceMetrics 계산을 검증합니다.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from src.core.event_bus import EventBus
from src.core.events import (
    BalanceUpdateEvent,
    BarEvent,
    FillEvent,
)
from src.eda.analytics import AnalyticsEngine, _freq_to_hours
from src.portfolio.cost_model import CostModel


def _make_balance(equity: float) -> BalanceUpdateEvent:
    return BalanceUpdateEvent(
        total_equity=equity,
        available_cash=equity * 0.5,
        total_margin_used=equity * 0.5,
        correlation_id=uuid4(),
        source="test",
    )


def _make_fill(
    symbol: str = "BTC/USDT",
    side: str = "BUY",
    price: float = 50000.0,
    qty: float = 0.1,
    fee: float = 1.0,
    ts: datetime | None = None,
) -> FillEvent:
    return FillEvent(
        client_order_id=f"test-{uuid4().hex[:8]}",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        fill_price=price,
        fill_qty=qty,
        fee=fee,
        fill_timestamp=ts or datetime.now(UTC),
        correlation_id=uuid4(),
        source="test",
    )


def _make_bar(
    symbol: str = "BTC/USDT",
    ts: datetime | None = None,
) -> BarEvent:
    return BarEvent(
        symbol=symbol,
        timeframe="1D",
        open=50000.0,
        high=51000.0,
        low=49000.0,
        close=50500.0,
        volume=1000.0,
        bar_timestamp=ts or datetime.now(UTC),
    )


class TestEquityCurve:
    """Equity curve 기록 테스트."""

    async def test_equity_curve_recorded(self) -> None:
        """BalanceUpdateEvent → equity curve 기록."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_balance(10000.0))
        await bus.publish(_make_balance(10500.0))
        await bus.publish(_make_balance(10300.0))
        await bus.stop()
        await task

        assert len(engine.equity_curve) == 3
        assert engine.equity_curve[0].equity == 10000.0
        assert engine.equity_curve[1].equity == 10500.0
        assert engine.equity_curve[2].equity == 10300.0

    async def test_equity_series_output(self) -> None:
        """get_equity_series() → pandas Series."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_balance(10000.0))
        await bus.publish(_make_balance(11000.0))
        await bus.stop()
        await task

        series = engine.get_equity_series()
        assert len(series) == 2
        assert series.iloc[0] == 10000.0
        assert series.iloc[1] == 11000.0


class TestTradeRecording:
    """Trade 기록 테스트."""

    async def test_round_trip_trade(self) -> None:
        """BUY → SELL round trip → TradeRecord 생성."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, tzinfo=UTC)
        # 진입: BUY 0.1 @ 50000
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=1.0, ts=t1))
        # 청산: SELL 0.1 @ 51000 → PnL = (51000-50000)*0.1 - 2.0 fees = 98.0
        await bus.publish(_make_fill(side="SELL", price=51000.0, qty=0.1, fee=1.0, ts=t2))
        await bus.stop()
        await task

        assert len(engine.closed_trades) == 1
        trade = engine.closed_trades[0]
        assert trade.direction == "LONG"
        assert trade.is_closed is True
        assert trade.is_profitable is True
        assert float(trade.pnl or 0) > 0

    async def test_short_trade(self) -> None:
        """SELL → BUY round trip (숏 거래)."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, tzinfo=UTC)
        # 진입: SELL 0.1 @ 50000
        await bus.publish(_make_fill(side="SELL", price=50000.0, qty=0.1, fee=1.0, ts=t1))
        # 청산: BUY 0.1 @ 49000 → PnL = (50000-49000)*0.1 - 2.0 = 98.0
        await bus.publish(_make_fill(side="BUY", price=49000.0, qty=0.1, fee=1.0, ts=t2))
        await bus.stop()
        await task

        assert len(engine.closed_trades) == 1
        trade = engine.closed_trades[0]
        assert trade.direction == "SHORT"
        assert trade.is_profitable is True

    async def test_unclosed_trade_not_recorded(self) -> None:
        """미종결 거래는 closed_trades에 포함되지 않음."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1))
        await bus.stop()
        await task

        assert len(engine.closed_trades) == 0
        assert engine.total_fills == 1


class TestPerformanceMetrics:
    """PerformanceMetrics 계산 테스트."""

    async def test_metrics_from_equity_curve(self) -> None:
        """Equity curve에서 기본 지표 계산."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        base = datetime(2024, 1, 1, tzinfo=UTC)
        # 10일간 equity curve
        for i in range(10):
            ts = base + timedelta(days=i)
            equity = 10000.0 + i * 100  # 단조 증가
            await bus.publish(_make_bar(ts=ts))
            await bus.publish(_make_balance(equity))
        await bus.stop()
        await task

        metrics = engine.compute_metrics()
        assert metrics.total_return > 0  # 양수 수익
        assert metrics.max_drawdown == 0.0  # 단조 증가 → MDD 0
        assert metrics.total_trades == 0  # 체결 없음

    async def test_metrics_with_trades(self) -> None:
        """거래 포함 시 win_rate, total_trades 계산."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, tzinfo=UTC)
        t3 = datetime(2024, 1, 3, tzinfo=UTC)
        t4 = datetime(2024, 1, 4, tzinfo=UTC)

        # Equity curve
        await bus.publish(_make_balance(10000.0))
        await bus.publish(_make_balance(10100.0))
        await bus.publish(_make_balance(10050.0))
        await bus.publish(_make_balance(10200.0))

        # 승리 거래
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0, ts=t1))
        await bus.publish(_make_fill(side="SELL", price=51000.0, qty=0.1, fee=0.0, ts=t2))
        # 패배 거래
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=0.0, ts=t3))
        await bus.publish(_make_fill(side="SELL", price=49000.0, qty=0.1, fee=0.0, ts=t4))

        await bus.stop()
        await task

        metrics = engine.compute_metrics()
        assert metrics.total_trades == 2
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 50.0


class TestBarTimestamps:
    """Bar timestamp dedup 테스트."""

    async def test_multi_symbol_dedup(self) -> None:
        """동일 timestamp의 멀티 심볼 bar는 한 번만 기록."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        await bus.publish(_make_bar(symbol="BTC/USDT", ts=ts))
        await bus.publish(_make_bar(symbol="ETH/USDT", ts=ts))
        await bus.stop()
        await task

        # 같은 timestamp → 1번만 기록
        assert engine.bar_count == 1


class TestFundingAdjustment:
    """H-002: 펀딩비 post-hoc 보정 테스트."""

    async def test_funding_adjustment_reduces_return(self) -> None:
        """펀딩비 보정 시 수익률이 하락."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(30):
            equity = 10000.0 + i * 50
            await bus.publish(_make_bar(ts=base + timedelta(days=i)))
            await bus.publish(_make_balance(equity))
        await bus.stop()
        await task

        # 펀딩비 없는 메트릭
        metrics_no_fund = engine.compute_metrics(timeframe="1D")
        # 펀딩비 있는 메트릭
        cost_model = CostModel(funding_rate_8h=0.0003)  # 높은 펀딩비
        metrics_fund = engine.compute_metrics(timeframe="1D", cost_model=cost_model)

        assert metrics_fund.total_return < metrics_no_fund.total_return


class TestTimeframeAwareness:
    """M-001: CAGR/Sharpe timeframe 인식 테스트."""

    def test_freq_to_hours(self) -> None:
        """_freq_to_hours 헬퍼 동작 확인."""
        assert _freq_to_hours("1D") == 24.0
        assert _freq_to_hours("4h") == 4.0
        assert _freq_to_hours("1h") == 1.0
        assert _freq_to_hours("15T") == 0.25

    async def test_cagr_4h_timeframe(self) -> None:
        """4h 타임프레임에서 CAGR이 올바르게 계산."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        base = datetime(2024, 1, 1, tzinfo=UTC)
        # 4h bar 100개 = 약 16.7일
        for i in range(100):
            equity = 10000.0 + i * 10
            ts = base + timedelta(hours=4 * i)
            await bus.publish(_make_bar(ts=ts))
            await bus.publish(_make_balance(equity))
        await bus.stop()
        await task

        metrics_1d = engine.compute_metrics(timeframe="1D")
        metrics_4h = engine.compute_metrics(timeframe="4h")

        # 4h bars 100개 = ~16.7일, 1D로 계산하면 100일로 착각 → CAGR 과소평가
        # 4h가 올바른 값 → CAGR이 더 높아야 함
        assert metrics_4h.cagr > metrics_1d.cagr

    async def test_sharpe_timeframe_aware(self) -> None:
        """1h 타임프레임에서 Sharpe가 1D보다 높아야 함 (더 많은 periods_per_year)."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        base = datetime(2024, 1, 1, tzinfo=UTC)
        for i in range(50):
            equity = 10000.0 + i * 20  # 단조 증가
            ts = base + timedelta(hours=i)
            await bus.publish(_make_bar(ts=ts))
            await bus.publish(_make_balance(equity))
        await bus.stop()
        await task

        sharpe_1d = engine.compute_metrics(timeframe="1D").sharpe_ratio
        sharpe_1h = engine.compute_metrics(timeframe="1h").sharpe_ratio

        # sqrt(8760) > sqrt(365) → 1h Sharpe > 1D Sharpe
        assert sharpe_1h > sharpe_1d


class TestWeightedAverageEntry:
    """M-002: 가중평균 진입가 테스트."""

    async def test_additional_buy_accumulates_size(self) -> None:
        """같은 방향 추가 매수 시 가중평균 진입가 + 수량 누적."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, tzinfo=UTC)
        t3 = datetime(2024, 1, 3, tzinfo=UTC)

        # 1차 매수: 0.1 @ 50000
        await bus.publish(_make_fill(side="BUY", price=50000.0, qty=0.1, fee=1.0, ts=t1))
        # 2차 매수: 0.1 @ 52000 (같은 방향 추가)
        await bus.publish(_make_fill(side="BUY", price=52000.0, qty=0.1, fee=1.0, ts=t2))
        # 청산: 0.2 @ 55000
        await bus.publish(_make_fill(side="SELL", price=55000.0, qty=0.2, fee=2.0, ts=t3))

        await bus.stop()
        await task

        assert len(engine.closed_trades) == 1
        trade = engine.closed_trades[0]
        # 가중평균 진입가: (50000*0.1 + 52000*0.1) / 0.2 = 51000
        assert float(trade.entry_price) == 51000.0
        assert float(trade.size) == 0.2
        # PnL = (55000 - 51000) * 0.2 - 4.0 fees = 796.0
        assert float(trade.pnl or 0) > 0


class TestEquityCurveNormalization:
    """M-003: equity curve bar 단위 정규화 테스트."""

    async def test_equity_curve_one_point_per_bar(self) -> None:
        """같은 timestamp의 여러 BalanceUpdate → 마지막 값만 유지."""
        engine = AnalyticsEngine(initial_capital=10000.0)
        bus = EventBus(queue_size=100)
        await engine.register(bus)

        task = asyncio.create_task(bus.start())
        ts = datetime(2024, 1, 1, tzinfo=UTC)

        # 같은 timestamp로 3번 업데이트
        b1 = BalanceUpdateEvent(
            total_equity=10000.0,
            available_cash=5000.0,
            total_margin_used=5000.0,
            timestamp=ts,
            correlation_id=uuid4(),
            source="test",
        )
        b2 = BalanceUpdateEvent(
            total_equity=10100.0,
            available_cash=5100.0,
            total_margin_used=5000.0,
            timestamp=ts,
            correlation_id=uuid4(),
            source="test",
        )
        b3 = BalanceUpdateEvent(
            total_equity=10200.0,
            available_cash=5200.0,
            total_margin_used=5000.0,
            timestamp=ts,
            correlation_id=uuid4(),
            source="test",
        )
        await bus.publish(b1)
        await bus.publish(b2)
        await bus.publish(b3)

        await bus.stop()
        await task

        # 같은 timestamp → 1개 포인트만 유지 (마지막 값)
        assert len(engine.equity_curve) == 1
        assert engine.equity_curve[0].equity == 10200.0
