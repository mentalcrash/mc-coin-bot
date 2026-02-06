"""EDA AnalyticsEngine.

모든 이벤트를 구독하여 실시간 메트릭을 수집하고,
최종 결과를 PerformanceMetrics / equity curve로 출력합니다.

Rules Applied:
    - Event-Driven 수집: BalanceUpdateEvent → equity curve, FillEvent → trade 기록
    - 기존 모델 재사용: PerformanceMetrics, TradeRecord
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    BarEvent,
    EventType,
    FillEvent,
)
from src.models.backtest import PerformanceMetrics, TradeRecord

# 최소 데이터 포인트 수 (Sharpe, drawdown 등 계산 가능 기준)
_MIN_DATA_POINTS = 2

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd

    from src.core.event_bus import EventBus


@dataclass
class EquityPoint:
    """Equity curve의 단일 데이터 포인트."""

    timestamp: datetime
    equity: float


@dataclass
class OpenTrade:
    """미종결 거래 추적용."""

    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    size: float
    fees: float = 0.0


class AnalyticsEngine:
    """EDA 분석 엔진.

    Subscribes to: BalanceUpdateEvent, FillEvent, BarEvent
    Collects: equity curve, trade records, bar timestamps

    Args:
        initial_capital: 초기 자본 (USD)
    """

    def __init__(self, initial_capital: float) -> None:
        self._initial_capital = initial_capital
        self._equity_curve: list[EquityPoint] = []
        self._closed_trades: list[TradeRecord] = []
        self._open_trades: dict[str, OpenTrade] = {}
        self._bar_timestamps: list[datetime] = []
        self._total_fills = 0

    async def register(self, bus: EventBus) -> None:
        """EventBus에 핸들러 등록."""
        bus.subscribe(EventType.BALANCE_UPDATE, self._on_balance_update)
        bus.subscribe(EventType.FILL, self._on_fill)
        bus.subscribe(EventType.BAR, self._on_bar)

    @property
    def equity_curve(self) -> list[EquityPoint]:
        """Equity curve 데이터."""
        return self._equity_curve

    @property
    def closed_trades(self) -> list[TradeRecord]:
        """종결된 거래 목록."""
        return self._closed_trades

    @property
    def total_fills(self) -> int:
        """총 체결 건수."""
        return self._total_fills

    @property
    def bar_count(self) -> int:
        """고유 bar timestamp 수."""
        return len(self._bar_timestamps)

    async def _on_balance_update(self, event: AnyEvent) -> None:
        """BalanceUpdateEvent → equity curve 기록."""
        assert isinstance(event, BalanceUpdateEvent)
        self._equity_curve.append(
            EquityPoint(timestamp=event.timestamp, equity=event.total_equity)
        )

    async def _on_fill(self, event: AnyEvent) -> None:
        """FillEvent → trade 기록."""
        assert isinstance(event, FillEvent)
        fill = event
        self._total_fills += 1

        symbol = fill.symbol
        is_buy = fill.side == "BUY"

        if symbol in self._open_trades:
            open_trade = self._open_trades[symbol]
            # 포지션 종료 조건: 반대 방향
            is_closing = (open_trade.direction == "LONG" and not is_buy) or (
                open_trade.direction == "SHORT" and is_buy
            )
            if is_closing:
                self._close_trade(open_trade, fill)
                return

        # 새 포지션 진입
        direction = "LONG" if is_buy else "SHORT"
        self._open_trades[symbol] = OpenTrade(
            symbol=symbol,
            direction=direction,
            entry_time=fill.fill_timestamp,
            entry_price=fill.fill_price,
            size=fill.fill_qty,
            fees=fill.fee,
        )

    def _close_trade(self, open_trade: OpenTrade, fill: FillEvent) -> None:
        """거래 종결 처리."""
        if open_trade.direction == "LONG":
            pnl = (fill.fill_price - open_trade.entry_price) * open_trade.size
        else:
            pnl = (open_trade.entry_price - fill.fill_price) * open_trade.size

        total_fees = open_trade.fees + fill.fee
        pnl_after_fees = pnl - total_fees
        entry_notional = open_trade.entry_price * open_trade.size
        pnl_pct = pnl_after_fees / entry_notional if entry_notional > 0 else 0.0

        trade = TradeRecord(
            entry_time=open_trade.entry_time,
            exit_time=fill.fill_timestamp,
            symbol=open_trade.symbol,
            direction=open_trade.direction,
            entry_price=Decimal(str(open_trade.entry_price)),
            exit_price=Decimal(str(fill.fill_price)),
            size=Decimal(str(open_trade.size)),
            pnl=Decimal(str(round(pnl_after_fees, 4))),
            pnl_pct=round(pnl_pct, 6),
            fees=Decimal(str(round(total_fees, 4))),
        )
        self._closed_trades.append(trade)
        del self._open_trades[open_trade.symbol]

    async def _on_bar(self, event: AnyEvent) -> None:
        """BarEvent → bar timestamp 기록."""
        assert isinstance(event, BarEvent)
        # 중복 timestamp 제거 (멀티 심볼)
        if not self._bar_timestamps or event.bar_timestamp != self._bar_timestamps[-1]:
            self._bar_timestamps.append(event.bar_timestamp)

    def get_equity_series(self) -> pd.Series:
        """Equity curve를 pandas Series로 반환."""
        import pandas as pd

        if not self._equity_curve:
            return pd.Series(dtype=float)

        timestamps = [p.timestamp for p in self._equity_curve]
        values = [p.equity for p in self._equity_curve]
        return pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float)

    def compute_metrics(self) -> PerformanceMetrics:
        """수집된 데이터로 PerformanceMetrics 계산."""
        trades = self._closed_trades
        winning = [t for t in trades if t.pnl is not None and t.pnl > 0]
        losing = [t for t in trades if t.pnl is not None and t.pnl <= 0]

        total_trades = len(trades)
        winning_count = len(winning)
        losing_count = len(losing)
        win_rate = (winning_count / total_trades * 100) if total_trades > 0 else 0.0

        # PnL 기반 지표
        avg_win = _avg_pnl_pct(winning) if winning else None
        avg_loss = _avg_pnl_pct(losing) if losing else None
        profit_factor = _profit_factor(winning, losing)

        # Equity curve 기반 지표
        equity_values = [p.equity for p in self._equity_curve]
        if len(equity_values) >= _MIN_DATA_POINTS:
            total_return = (equity_values[-1] / equity_values[0] - 1) * 100
            returns = np.diff(equity_values) / np.array(equity_values[:-1])
            sharpe = _annualized_sharpe(returns)
            max_dd = _max_drawdown(equity_values)
            cagr = _compute_cagr(equity_values, len(self._bar_timestamps))
            vol = float(np.std(returns) * np.sqrt(365) * 100) if len(returns) > 1 else None
        else:
            total_return = 0.0
            sharpe = 0.0
            max_dd = 0.0
            cagr = 0.0
            vol = None

        return PerformanceMetrics(
            total_return=round(total_return, 2),
            cagr=round(cagr, 2),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 2),
            win_rate=round(win_rate, 2),
            total_trades=total_trades,
            winning_trades=winning_count,
            losing_trades=losing_count,
            avg_win=round(avg_win, 4) if avg_win is not None else None,
            avg_loss=round(avg_loss, 4) if avg_loss is not None else None,
            profit_factor=round(profit_factor, 4) if profit_factor is not None else None,
            volatility=round(vol, 2) if vol is not None else None,
        )


# ==========================================================================
# Helper functions
# ==========================================================================
def _avg_pnl_pct(trades: list[TradeRecord]) -> float:
    """평균 PnL %."""
    pcts = [t.pnl_pct for t in trades if t.pnl_pct is not None]
    return float(np.mean(pcts) * 100) if pcts else 0.0


def _profit_factor(
    winning: list[TradeRecord],
    losing: list[TradeRecord],
) -> float | None:
    """Profit factor = gross profit / gross loss."""
    gross_profit = sum(float(t.pnl) for t in winning if t.pnl is not None)
    gross_loss = abs(sum(float(t.pnl) for t in losing if t.pnl is not None))
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def _annualized_sharpe(returns: np.ndarray) -> float:  # type: ignore[type-arg]
    """연환산 Sharpe ratio (365일 기준)."""
    if len(returns) < _MIN_DATA_POINTS:
        return 0.0
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1))
    if std_ret == 0:
        return 0.0
    return mean_ret / std_ret * np.sqrt(365)


def _max_drawdown(equity_values: list[float]) -> float:
    """최대 낙폭 (%)."""
    peak = equity_values[0]
    max_dd = 0.0
    for val in equity_values:
        peak = max(peak, val)
        dd = (peak - val) / peak * 100
        max_dd = max(max_dd, dd)
    return max_dd


def _compute_cagr(equity_values: list[float], n_bars: int) -> float:
    """CAGR (연평균 복리 수익률, %)."""
    if n_bars <= 0 or equity_values[0] <= 0:
        return 0.0
    years = n_bars / 365
    if years <= 0:
        return 0.0
    total_return = equity_values[-1] / equity_values[0]
    return (total_return ** (1 / years) - 1) * 100
