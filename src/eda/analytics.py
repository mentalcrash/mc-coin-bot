"""EDA AnalyticsEngine.

모든 이벤트를 구독하여 실시간 메트릭을 수집하고,
최종 결과를 PerformanceMetrics / equity curve로 출력합니다.

Rules Applied:
    - Event-Driven 수집: BalanceUpdateEvent → equity curve, FillEvent → trade 기록
    - 기존 모델 재사용: PerformanceMetrics, TradeRecord
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from src.backtest.metrics import build_performance_metrics, freq_to_periods_per_year
from src.core.events import (
    AnyEvent,
    BalanceUpdateEvent,
    BarEvent,
    EventType,
    FillEvent,
)
from src.models.backtest import PerformanceMetrics, TradeRecord

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd

    from src.core.event_bus import EventBus
    from src.portfolio.cost_model import CostModel


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
        # M-003: bar 단위 equity 정규화 — 같은 bar 내 마지막 업데이트만 유지
        self._last_equity_ts: datetime | None = None

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
        """BalanceUpdateEvent → equity curve 기록.

        M-003: 같은 bar timestamp 내 여러 업데이트가 오면 마지막 값으로 덮어쓰기.
        (한 bar에 Signal→Fill→BalanceUpdate가 여러 번 발생할 수 있음)
        """
        assert isinstance(event, BalanceUpdateEvent)
        ts = event.timestamp
        if self._last_equity_ts is not None and ts == self._last_equity_ts and self._equity_curve:
            # 같은 timestamp → 마지막 값 덮어쓰기
            self._equity_curve[-1] = EquityPoint(timestamp=ts, equity=event.total_equity)
        else:
            self._equity_curve.append(EquityPoint(timestamp=ts, equity=event.total_equity))
        self._last_equity_ts = ts

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

            # M-002: 같은 방향 추가 매매 → 가중평균 진입가 + 수량 누적
            same_direction = (open_trade.direction == "LONG" and is_buy) or (
                open_trade.direction == "SHORT" and not is_buy
            )
            if same_direction:
                new_size = open_trade.size + fill.fill_qty
                if new_size > 0:
                    open_trade.entry_price = (
                        open_trade.entry_price * open_trade.size + fill.fill_price * fill.fill_qty
                    ) / new_size
                    open_trade.size = new_size
                    open_trade.fees += fill.fee
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
        # open trade 삭제를 먼저 보장 (ValidationError 시 cascade 방지)
        del self._open_trades[open_trade.symbol]

        if open_trade.direction == "LONG":
            pnl = (fill.fill_price - open_trade.entry_price) * open_trade.size
        else:
            pnl = (open_trade.entry_price - fill.fill_price) * open_trade.size

        total_fees = open_trade.fees + fill.fee
        pnl_after_fees = pnl - total_fees
        entry_notional = open_trade.entry_price * open_trade.size
        pnl_pct = pnl_after_fees / entry_notional if entry_notional > 0 else 0.0

        # NaN/Inf 방어: 유한하지 않은 값은 0으로 대체
        if not math.isfinite(open_trade.entry_price) or not math.isfinite(open_trade.size):
            return
        if not math.isfinite(pnl_after_fees):
            pnl_after_fees = 0.0
        if not math.isfinite(total_fees):
            total_fees = 0.0

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

    def get_strategy_returns(
        self,
        timeframe: str = "1D",
        cost_model: CostModel | None = None,
    ) -> pd.Series:
        """Funding drag 적용된 리샘플링 수익률 시리즈 반환.

        compute_metrics()와 QuantStats 리포트 양쪽에서 동일한 데이터를 사용하기 위해
        공통 메서드로 추출.

        Args:
            timeframe: 데이터 주기 (예: "1D", "4h")
            cost_model: 비용 모델 (펀딩비 보정)

        Returns:
            리샘플링 + funding drag 적용된 수익률 시리즈
        """
        import pandas as pd

        equity_series = self.get_equity_series()
        if len(equity_series) == 0:
            return pd.Series(dtype=float)

        pandas_freq = tf_to_pandas_freq(timeframe)
        equity_resampled = equity_series.resample(pandas_freq).last().dropna()
        returns: pd.Series = equity_resampled.pct_change().dropna()  # type: ignore[assignment]

        # H-002: 펀딩비 post-hoc 보정
        ppy = freq_to_periods_per_year(timeframe)
        if cost_model is not None and cost_model.funding_rate_8h > 0:
            hours_per_bar = _HOURS_PER_YEAR / ppy
            funding_drag = cost_model.funding_rate_8h * (hours_per_bar / 8.0)
            returns = returns - funding_drag

        return returns

    def compute_metrics(
        self,
        timeframe: str = "1D",
        cost_model: CostModel | None = None,
    ) -> PerformanceMetrics:
        """수집된 데이터로 PerformanceMetrics 계산.

        Args:
            timeframe: 데이터 주기 (M-001: Sharpe/CAGR 연환산에 사용)
            cost_model: 비용 모델 (H-002: 펀딩비 post-hoc 보정에 사용)
        """
        equity_series = self.get_equity_series()

        # Equity curve를 target TF로 리샘플링 (intrabar 업데이트 → 일별 등)
        if len(equity_series) > 0:
            pandas_freq = tf_to_pandas_freq(timeframe)
            equity_series = equity_series.resample(pandas_freq).last().dropna()

        ppy = freq_to_periods_per_year(timeframe)

        # H-002: 펀딩비 post-hoc 보정 (per equity-period drag)
        funding_drag = 0.0
        if cost_model is not None and cost_model.funding_rate_8h > 0:
            hours_per_bar = _HOURS_PER_YEAR / ppy
            funding_drag = cost_model.funding_rate_8h * (hours_per_bar / 8.0)

        return build_performance_metrics(
            equity_curve=equity_series,
            trades=self._closed_trades,
            periods_per_year=ppy,
            risk_free_rate=0.0,
            funding_drag_per_period=funding_drag,
        )


_HOURS_PER_YEAR = 365 * 24

# Timeframe → pandas resample frequency 매핑
TF_FREQ_MAP: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1D": "1D",
    "1d": "1D",
}


def tf_to_pandas_freq(tf: str) -> str:
    """Target timeframe → pandas resample frequency."""
    return TF_FREQ_MAP.get(tf, tf)
