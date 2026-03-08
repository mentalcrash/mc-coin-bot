"""Health Check 알림 데이터 모델.

HealthCheckScheduler에서 수집한 시스템/마켓/전략 상태를 표현하는
Pydantic frozen 모델입니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, ConfigDict
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SystemHealthSnapshot(BaseModel):
    """Tier 1: System Heartbeat 데이터.

    Attributes:
        timestamp: 수집 시각
        uptime_seconds: 봇 가동 시간 (초)
        total_equity: 총 자산
        available_cash: 가용 현금
        capital_utilization: 자본 활용률
        open_position_count: 오픈 포지션 수
        total_symbols: 전체 심볼 수
        current_drawdown: 현재 drawdown (0.0~1.0)
        peak_equity: 고점 equity
        is_circuit_breaker_active: CB 발동 여부
        today_pnl: 금일 실현 PnL
        today_trades: 금일 거래 건수
        stale_symbol_count: stale 심볼 수
        bars_emitted: 발행된 bar 이벤트 수
        events_dropped: 드롭된 이벤트 수
        max_queue_depth: 최대 큐 깊이
        is_notification_degraded: 알림 시스템 degraded 여부
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    uptime_seconds: float
    total_equity: float
    available_cash: float
    capital_utilization: float
    open_position_count: int
    total_symbols: int
    current_drawdown: float
    peak_equity: float
    is_circuit_breaker_active: bool
    today_pnl: float
    today_trades: int
    stale_symbol_count: int
    bars_emitted: int
    events_dropped: int
    max_queue_depth: int
    is_notification_degraded: bool
    safety_stop_count: int = 0
    safety_stop_failures: int = 0


class PositionStatus(BaseModel):
    """오픈 포지션 요약.

    Attributes:
        symbol: 심볼
        direction: 방향 (LONG/SHORT)
        unrealized_pnl: 미실현 PnL
        size: 포지션 수량
        current_weight: 포트폴리오 비중
    """

    model_config = ConfigDict(frozen=True)

    symbol: str
    direction: str
    unrealized_pnl: float
    size: float
    current_weight: float


class StrategyPerformanceSnapshot(BaseModel):
    """개별 전략 성과 스냅샷 (8h 리포트 전략별 breakdown).

    Attributes:
        strategy_name: 전략 이름
        rolling_sharpe: 최근 30일 rolling Sharpe
        win_rate: win rate
        total_pnl: 누적 PnL
        trade_count: 거래 건수
        status: 전략 상태 아이콘 (HEALTHY/WATCH/DEGRADING)
    """

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    rolling_sharpe: float
    win_rate: float
    total_pnl: float
    trade_count: int
    status: str


class AssetDashboardItem(BaseModel):
    """에셋별 대시보드 데이터 (Daily Report Section 3)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    signal: str  # "LONG" / "NEUTRAL"
    current_price: float
    change_24h_pct: float
    position_value: float
    day_pnl: float
    stop_price: float | None = None
    stop_distance_pct: float | None = None


class StrategyIndicatorItem(BaseModel):
    """에셋별 전략 지표 (Daily Report Section 4)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    supertrend_line: float | None = None
    adx_value: float | None = None
    outlook: str = ""


class StrategyInfoMixin(BaseModel):
    """Strategy Info 공통 필드."""

    model_config = ConfigDict(frozen=True)

    strategy_name: str
    strategy_params: dict[str, str]
    trailing_stop_config: str
    timeframe: str


class SystemHealthMixin(BaseModel):
    """System Health 공통 필드."""

    model_config = ConfigDict(frozen=True)

    uptime_seconds: float
    is_circuit_breaker_active: bool
    ws_ok_count: int
    ws_total_count: int
    rolling_sharpe_30d: float
    win_rate: float
    profit_factor: float
    alpha_decay_detected: bool


class DailyReportData(StrategyInfoMixin):
    """Spot Daily Report 전체 데이터 (5 sections)."""

    model_config = ConfigDict(frozen=True)

    # Section 2: Portfolio Summary
    total_equity: float
    available_cash: float
    cash_pct: float
    today_pnl: float
    invested_count: int
    total_asset_count: int
    cumulative_return_pct: float
    max_drawdown_pct: float
    rolling_sharpe_30d: float

    # Section 3: Asset Dashboard
    assets: tuple[AssetDashboardItem, ...]

    # Section 4: Strategy Indicators
    indicators: tuple[StrategyIndicatorItem, ...]

    # Section 5: System Health (partial — no rolling_sharpe_30d in mixin)
    uptime_seconds: float
    is_circuit_breaker_active: bool
    ws_ok_count: int
    ws_total_count: int
    win_rate: float
    profit_factor: float
    alpha_decay_detected: bool


class SignalChangeItem(BaseModel):
    """시그널 변동 항목 (Bar Close Report)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    prev_signal: str  # "LONG" / "NEUTRAL"
    new_signal: str
    realized_pnl: float | None = None  # 청산 시 실현 PnL


class BarCloseReportData(BaseModel):
    """12H Bar Close Report 데이터."""

    model_config = ConfigDict(frozen=True)

    bar_time_utc: str  # "00:00" / "12:00"

    # Section 1: Signal Changes
    signal_changes: tuple[SignalChangeItem, ...]

    # Section 2: Asset Dashboard (간소화)
    assets: tuple[AssetDashboardItem, ...]

    # Section 3: Portfolio Snapshot
    total_equity: float
    available_cash: float
    capital_deployed: float
    drawdown_pct: float
    capital_utilization: float
    today_pnl: float
    invested_count: int
    total_asset_count: int

    # Section 4: System Status
    uptime_seconds: float
    is_circuit_breaker_active: bool
    ws_ok_count: int
    ws_total_count: int


class AssetWeeklyPerformance(BaseModel):
    """에셋별 주간 성과 (Weekly Report Section 3)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    signal: str  # "LONG" / "NEUTRAL"
    current_price: float
    week_change_pct: float
    week_pnl: float
    week_trades: int


class WeeklyReportData(StrategyInfoMixin, SystemHealthMixin):
    """Spot Weekly Report 전체 데이터 (6 sections)."""

    model_config = ConfigDict(frozen=True)

    # Section 2: Weekly Portfolio Summary
    total_equity: float
    available_cash: float
    cash_pct: float
    week_pnl: float
    week_trades: int
    invested_count: int
    total_asset_count: int
    cumulative_return_pct: float
    max_drawdown_pct: float

    # Section 3: Asset Weekly Performance
    assets: tuple[AssetWeeklyPerformance, ...]

    # Section 4: Weekly Trade Summary
    best_trade_symbol: str
    best_trade_pnl: float
    worst_trade_symbol: str
    worst_trade_pnl: float
    week_win_rate: float
    week_profit_factor: float

    # Section 5: Strategy Indicators
    indicators: tuple[StrategyIndicatorItem, ...]


class AssetMonthlyPerformance(BaseModel):
    """에셋별 월간 성과 (Monthly Report Section 3)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    signal: str  # "LONG" / "NEUTRAL"
    current_price: float
    month_change_pct: float
    month_pnl: float
    month_trades: int


class MonthlyPerformanceTrend(BaseModel):
    """월별 성과 추이 항목 (Monthly Report Section 5)."""

    model_config = ConfigDict(frozen=True)

    year_month: str  # "2026-03"
    pnl: float
    return_pct: float
    trades: int
    sharpe: float


class MonthlyReportData(StrategyInfoMixin, SystemHealthMixin):
    """Spot Monthly Report 전체 데이터 (8 sections)."""

    model_config = ConfigDict(frozen=True)

    # Section 2: Monthly Portfolio Summary
    total_equity: float
    available_cash: float
    cash_pct: float
    month_pnl: float
    month_trades: int
    month_return_pct: float
    invested_count: int
    total_asset_count: int
    cumulative_return_pct: float
    max_drawdown_pct: float

    # Section 3: Asset Monthly Performance
    assets: tuple[AssetMonthlyPerformance, ...]

    # Section 4: Monthly Trade Summary
    best_trade_symbol: str
    best_trade_pnl: float
    worst_trade_symbol: str
    worst_trade_pnl: float
    month_win_rate: float
    month_profit_factor: float
    avg_trade_pnl: float
    total_fees: float

    # Section 5: Performance Trend (최근 3개월)
    performance_trend: tuple[MonthlyPerformanceTrend, ...]

    # Section 6: Strategy Indicators
    indicators: tuple[StrategyIndicatorItem, ...]

    # Section 8: Risk Summary
    month_max_drawdown_pct: float
    longest_losing_streak: int


class AssetQuarterlyPerformance(BaseModel):
    """에셋별 분기 성과 (Quarterly Report Section 3)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    signal: str
    current_price: float
    quarter_change_pct: float
    quarter_pnl: float
    quarter_trades: int


class QuarterlyReportData(StrategyInfoMixin, SystemHealthMixin):
    """Spot Quarterly Report 전체 데이터 (8 sections)."""

    model_config = ConfigDict(frozen=True)

    # Section 2: Quarterly Portfolio Summary
    total_equity: float
    available_cash: float
    cash_pct: float
    quarter_pnl: float
    quarter_trades: int
    quarter_return_pct: float
    invested_count: int
    total_asset_count: int
    cumulative_return_pct: float
    max_drawdown_pct: float

    # Section 3: Asset Quarterly Performance
    assets: tuple[AssetQuarterlyPerformance, ...]

    # Section 4: Quarterly Trade Summary
    best_trade_symbol: str
    best_trade_pnl: float
    worst_trade_symbol: str
    worst_trade_pnl: float
    quarter_win_rate: float
    quarter_profit_factor: float
    avg_trade_pnl: float
    total_fees: float

    # Section 5: Monthly Performance Trend (3개월)
    performance_trend: tuple[MonthlyPerformanceTrend, ...]

    # Section 6: Strategy Indicators
    indicators: tuple[StrategyIndicatorItem, ...]

    # Section 8: Risk Summary
    quarter_max_drawdown_pct: float
    longest_losing_streak: int


class AssetYearlyPerformance(BaseModel):
    """에셋별 연간 성과 (Yearly Report Section 3)."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    signal: str
    current_price: float
    year_change_pct: float
    year_pnl: float
    year_trades: int


class QuarterlyPerformanceTrend(BaseModel):
    """분기별 성과 추이 항목 (Yearly Report Section 5)."""

    model_config = ConfigDict(frozen=True)

    year_quarter: str  # "2026-Q1"
    pnl: float
    return_pct: float
    trades: int
    sharpe: float


class YearlyReportData(StrategyInfoMixin, SystemHealthMixin):
    """Spot Yearly Report 전체 데이터 (9 sections)."""

    model_config = ConfigDict(frozen=True)

    # Section 2: Yearly Portfolio Summary
    total_equity: float
    available_cash: float
    cash_pct: float
    year_pnl: float
    year_trades: int
    year_return_pct: float
    invested_count: int
    total_asset_count: int
    cumulative_return_pct: float
    max_drawdown_pct: float

    # Section 3: Asset Yearly Performance
    assets: tuple[AssetYearlyPerformance, ...]

    # Section 4: Yearly Trade Summary
    best_trade_symbol: str
    best_trade_pnl: float
    worst_trade_symbol: str
    worst_trade_pnl: float
    year_win_rate: float
    year_profit_factor: float
    avg_trade_pnl: float
    total_fees: float

    # Section 5: Quarterly Performance Trend (4분기)
    quarterly_trend: tuple[QuarterlyPerformanceTrend, ...]

    # Section 6: Monthly Performance Trend (12개월)
    monthly_trend: tuple[MonthlyPerformanceTrend, ...]

    # Section 7: Strategy Indicators
    indicators: tuple[StrategyIndicatorItem, ...]

    # Section 9: Risk Summary
    year_max_drawdown_pct: float
    longest_losing_streak: int


class StrategyHealthSnapshot(BaseModel):
    """Tier 3: Strategy Health 리포트.

    Attributes:
        timestamp: 수집 시각
        rolling_sharpe_30d: 최근 30일 rolling Sharpe
        win_rate_recent: 최근 20건 win rate
        profit_factor: 최근 20건 profit factor
        total_closed_trades: 전체 종결 거래 수
        open_positions: 오픈 포지션 목록
        is_circuit_breaker_active: CB 발동 여부
        alpha_decay_detected: 3연속 Sharpe 하락 감지
        strategy_breakdown: 전략별 성과 스냅샷
    """

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    rolling_sharpe_30d: float
    win_rate_recent: float
    profit_factor: float
    total_closed_trades: int
    open_positions: tuple[PositionStatus, ...]
    is_circuit_breaker_active: bool
    alpha_decay_detected: bool
    strategy_breakdown: tuple[StrategyPerformanceSnapshot, ...] = ()
