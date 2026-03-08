"""Event -> Discord Embed 변환 함수.

각 EDA 이벤트 타입을 Discord Embed dict로 포맷팅합니다.
DiscordNotifier의 기존 embed 스타일(색상, 필드 레이아웃)을 따릅니다.

Rules Applied:
    - #10 Python Standards: Pure functions, type hints
    - #22 Notification Standards: Rich Embeds, color segmentation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.events import (
        BalanceUpdateEvent,
        CircuitBreakerEvent,
        FillEvent,
        PositionUpdateEvent,
        RiskAlertEvent,
    )
    from src.models.backtest import PerformanceMetrics, TradeRecord
    from src.notification.health_models import (
        BarCloseReportData,
        DailyReportData,
        MonthlyReportData,
        QuarterlyReportData,
        StrategyHealthSnapshot,
        StrategyIndicatorItem,
        StrategyInfoMixin,
        SystemHealthMixin,
        SystemHealthSnapshot,
        WeeklyReportData,
        YearlyReportData,
    )

# Discord Embed 색상 코드 (decimal)
_COLOR_GREEN = 0x57F287
_COLOR_RED = 0xED4245
_COLOR_BLUE = 0x3498DB
_COLOR_YELLOW = 0xFFFF00
_COLOR_ORANGE = 0xE67E22

_FOOTER_TEXT = "MC-Coin-Bot"


def _fmt_price(value: float) -> str:
    """가격을 사람이 읽기 좋은 형식으로 포맷.

    $1 이상: 소수점 2자리 (예: $67,260.00)
    $0.01~$1: 소수점 4자리 (예: $0.0386)
    $0.01 미만: 유효숫자 4개 (예: $0.001234)
    """
    abs_val = abs(value)
    if abs_val >= 1:
        return f"${value:,.2f}"
    if abs_val >= 0.01:
        return f"${value:,.4f}"
    return f"${value:.4g}"


def format_fill_embed(fill: FillEvent) -> dict[str, Any]:
    """FillEvent -> Discord Embed dict (position 정보 없을 때 fallback).

    Args:
        fill: 체결 이벤트

    Returns:
        Discord Embed dict
    """
    is_buy = fill.side == "BUY"
    color = _COLOR_GREEN if is_buy else _COLOR_RED
    side_label = "BUY" if is_buy else "SELL"
    value = fill.fill_price * fill.fill_qty

    return {
        "title": f"{side_label}: {fill.symbol}",
        "color": color,
        "fields": [
            {"name": "Price", "value": f"${fill.fill_price:,.4f}", "inline": True},
            {"name": "Qty", "value": f"{fill.fill_qty:,.6f}", "inline": True},
            {"name": "Fee", "value": f"${fill.fee:,.4f}", "inline": True},
            {"name": "Value", "value": f"${value:,.2f}", "inline": True},
        ],
        "timestamp": fill.fill_timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_fill_with_position_embed(
    fill: FillEvent, position: PositionUpdateEvent
) -> dict[str, Any]:
    """FillEvent + PositionUpdateEvent -> 통합 Discord Embed dict.

    체결 정보와 결과 포지션 상태를 하나의 알림으로 표시합니다.

    Args:
        fill: 체결 이벤트
        position: 포지션 업데이트 이벤트

    Returns:
        Discord Embed dict
    """
    is_buy = fill.side == "BUY"
    color = _COLOR_GREEN if is_buy else _COLOR_RED
    side_label = "BUY" if is_buy else "SELL"
    value = fill.fill_price * fill.fill_qty

    return {
        "title": f"{side_label}: {fill.symbol}",
        "color": color,
        "fields": [
            {"name": "Price", "value": f"${fill.fill_price:,.4f}", "inline": True},
            {"name": "Qty", "value": f"{fill.fill_qty:,.6f}", "inline": True},
            {"name": "Fee", "value": f"${fill.fee:,.4f}", "inline": True},
            {"name": "Value", "value": f"${value:,.2f}", "inline": True},
            {
                "name": "Position",
                "value": f"{position.direction.name} {position.size:,.6f}",
                "inline": True,
            },
            {
                "name": "Avg Entry",
                "value": f"${position.avg_entry_price:,.4f}",
                "inline": True,
            },
            {
                "name": "Realized PnL",
                "value": f"${position.realized_pnl:+,.2f}",
                "inline": True,
            },
        ],
        "timestamp": fill.fill_timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_circuit_breaker_embed(event: CircuitBreakerEvent) -> dict[str, Any]:
    """CircuitBreakerEvent -> Discord Embed dict.

    Args:
        event: 서킷 브레이커 이벤트

    Returns:
        Discord Embed dict
    """
    close_label = "Yes" if event.close_all_positions else "No"

    return {
        "title": "CIRCUIT BREAKER TRIGGERED",
        "color": _COLOR_ORANGE,
        "description": event.reason,
        "fields": [
            {"name": "Close All Positions", "value": close_label, "inline": True},
        ],
        "timestamp": event.timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_risk_alert_embed(event: RiskAlertEvent) -> dict[str, Any]:
    """RiskAlertEvent -> Discord Embed dict.

    Args:
        event: 리스크 경고 이벤트

    Returns:
        Discord Embed dict
    """
    is_critical = event.alert_level == "CRITICAL"
    color = _COLOR_ORANGE if is_critical else _COLOR_YELLOW

    return {
        "title": f"RISK {event.alert_level}",
        "color": color,
        "description": event.message,
        "fields": [
            {"name": "Level", "value": event.alert_level, "inline": True},
        ],
        "timestamp": event.timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_balance_embed(event: BalanceUpdateEvent) -> dict[str, Any]:
    """BalanceUpdateEvent -> Discord Embed dict.

    Args:
        event: 잔고 업데이트 이벤트

    Returns:
        Discord Embed dict
    """
    return {
        "title": "Balance Update",
        "color": _COLOR_BLUE,
        "fields": [
            {"name": "Total Equity", "value": f"${event.total_equity:,.2f}", "inline": True},
            {"name": "Available Cash", "value": f"${event.available_cash:,.2f}", "inline": True},
            {
                "name": "Capital Deployed",
                "value": f"${event.capital_deployed:,.2f}",
                "inline": True,
            },
        ],
        "timestamp": event.timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_position_embed(event: PositionUpdateEvent) -> dict[str, Any]:
    """PositionUpdateEvent -> Discord Embed dict.

    Args:
        event: 포지션 업데이트 이벤트

    Returns:
        Discord Embed dict
    """
    direction_label = event.direction.name
    color = _COLOR_GREEN if event.unrealized_pnl >= 0 else _COLOR_RED

    return {
        "title": f"Position: {event.symbol}",
        "color": color,
        "fields": [
            {"name": "Direction", "value": direction_label, "inline": True},
            {"name": "Size", "value": f"{event.size:,.6f}", "inline": True},
            {
                "name": "Avg Entry",
                "value": f"${event.avg_entry_price:,.4f}",
                "inline": True,
            },
            {
                "name": "Unrealized PnL",
                "value": f"${event.unrealized_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Realized PnL",
                "value": f"${event.realized_pnl:+,.2f}",
                "inline": True,
            },
        ],
        "timestamp": event.timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_daily_report_embed(
    metrics: PerformanceMetrics,
    open_positions: int,
    total_equity: float,
    trades_today: list[TradeRecord],
) -> dict[str, Any]:
    """Daily report → Discord Embed dict.

    Args:
        metrics: 현재 성과 지표
        open_positions: 오픈 포지션 수
        total_equity: 현재 equity
        trades_today: 오늘 거래 목록

    Returns:
        Discord Embed dict
    """
    total_pnl = sum(float(t.pnl) for t in trades_today if t.pnl is not None)

    return {
        "title": "Daily Report",
        "color": _COLOR_BLUE,
        "fields": [
            {"name": "Today's Trades", "value": str(len(trades_today)), "inline": True},
            {"name": "Today's PnL", "value": f"${total_pnl:+,.2f}", "inline": True},
            {"name": "Total Equity", "value": f"${total_equity:,.2f}", "inline": True},
            {"name": "Max Drawdown", "value": f"{metrics.max_drawdown:.2f}%", "inline": True},
            {"name": "Open Positions", "value": str(open_positions), "inline": True},
            {"name": "Sharpe Ratio", "value": f"{metrics.sharpe_ratio:.2f}", "inline": True},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_weekly_report_embed(
    metrics: PerformanceMetrics,
    trades_week: list[TradeRecord],
) -> dict[str, Any]:
    """Weekly report → Discord Embed dict.

    Args:
        metrics: 현재 성과 지표
        trades_week: 이번 주 거래 목록

    Returns:
        Discord Embed dict
    """
    total_pnl = sum(float(t.pnl) for t in trades_week if t.pnl is not None)
    pnls = [float(t.pnl) for t in trades_week if t.pnl is not None]
    best = max(pnls) if pnls else 0.0
    worst = min(pnls) if pnls else 0.0

    return {
        "title": "Weekly Report",
        "color": _COLOR_BLUE,
        "fields": [
            {"name": "Weekly Trades", "value": str(len(trades_week)), "inline": True},
            {"name": "Weekly PnL", "value": f"${total_pnl:+,.2f}", "inline": True},
            {"name": "Sharpe Ratio", "value": f"{metrics.sharpe_ratio:.2f}", "inline": True},
            {"name": "Max Drawdown", "value": f"{metrics.max_drawdown:.2f}%", "inline": True},
            {"name": "Best Trade", "value": f"${best:+,.2f}", "inline": True},
            {"name": "Worst Trade", "value": f"${worst:+,.2f}", "inline": True},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_safety_stop_failure_embed(symbol: str, failure_count: int) -> dict[str, Any]:
    """Safety stop 연속 실패 CRITICAL 알림 embed.

    Args:
        symbol: 대상 심볼
        failure_count: 연속 실패 횟수

    Returns:
        Discord Embed dict
    """
    return {
        "title": "SAFETY STOP FAILURE",
        "color": _COLOR_RED,
        "description": (
            f"**{symbol}** safety stop placement failed **{failure_count}** consecutive times.\n"
            "Exchange safety net may be **INACTIVE** — manual intervention required."
        ),
        "fields": [
            {"name": "Symbol", "value": symbol, "inline": True},
            {"name": "Failures", "value": str(failure_count), "inline": True},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_safety_stop_stale_embed(symbol: str) -> dict[str, Any]:
    """Stale safety stop 감지 WARNING 알림 embed.

    재시작 후 거래소에 해당 주문이 존재하지 않을 때 발행합니다.

    Args:
        symbol: 대상 심볼

    Returns:
        Discord Embed dict
    """
    return {
        "title": "SAFETY STOP STALE",
        "color": _COLOR_ORANGE,
        "description": (
            f"**{symbol}** safety stop was restored from state "
            "but not found on exchange. Will be re-placed on next bar."
        ),
        "fields": [
            {"name": "Symbol", "value": symbol, "inline": True},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_stop_ratchet_embed(symbol: str, old_stop: float, new_stop: float) -> dict[str, Any]:
    """Stop-Limit Ratchet (stop price 상향) 알림.

    Args:
        symbol: 대상 심볼
        old_stop: 이전 stop price
        new_stop: 새 stop price

    Returns:
        Discord Embed dict
    """
    change_pct = ((new_stop - old_stop) / old_stop * 100) if old_stop > 0 else 0.0
    return {
        "title": f"Stop Ratchet: {symbol}",
        "color": _COLOR_GREEN,
        "fields": [
            {"name": "Old Stop", "value": f"${old_stop:,.2f}", "inline": True},
            {"name": "New Stop", "value": f"${new_stop:,.2f}", "inline": True},
            {"name": "Change", "value": f"+{change_pct:.2f}%", "inline": True},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_enhanced_daily_report_embed(
    metrics: PerformanceMetrics,
    open_positions: int,
    total_equity: float,
    trades_today: list[TradeRecord],
    system_health: SystemHealthSnapshot | None = None,
    strategy_health: StrategyHealthSnapshot | None = None,
) -> dict[str, Any]:
    """Enhanced Daily Report -> Discord Embed dict.

    기본 Daily Report에 시스템/전략 건강 데이터를 통합합니다.
    health 데이터가 None이면 기본 Daily Report와 동일한 결과를 반환합니다.

    Args:
        metrics: 현재 성과 지표
        open_positions: 오픈 포지션 수
        total_equity: 현재 equity
        trades_today: 오늘 거래 목록
        system_health: 시스템 건강 스냅샷 (None이면 생략)
        strategy_health: 전략 건강 스냅샷 (None이면 생략)

    Returns:
        Discord Embed dict
    """
    total_pnl = sum(float(t.pnl) for t in trades_today if t.pnl is not None)

    # Alpha decay 감지 시 RED, 그 외 BLUE
    alpha_decay = strategy_health.alpha_decay_detected if strategy_health else False
    color = _COLOR_RED if alpha_decay else _COLOR_BLUE

    fields: list[dict[str, Any]] = [
        # Performance (inline 3)
        {"name": "Today's PnL", "value": f"${total_pnl:+,.2f}", "inline": True},
        {"name": "Total Equity", "value": f"${total_equity:,.2f}", "inline": True},
        {"name": "Max Drawdown", "value": f"{metrics.max_drawdown:.2f}%", "inline": True},
        # Stats (inline 3)
        {"name": "Sharpe Ratio", "value": f"{metrics.sharpe_ratio:.2f}", "inline": True},
        {"name": "Open Positions", "value": str(open_positions), "inline": True},
        {"name": "Today's Trades", "value": str(len(trades_today)), "inline": True},
    ]

    # Strategy Breakdown (non-inline 1)
    if strategy_health and strategy_health.strategy_breakdown:
        _status_icons = {"HEALTHY": "+", "WATCH": "~", "DEGRADING": "-"}
        breakdown_lines: list[str] = []
        for sp in strategy_health.strategy_breakdown:
            icon = _status_icons.get(sp.status, "?")
            line = (
                f"[{icon}] **{sp.strategy_name}** | "
                + f"Sharpe {sp.rolling_sharpe:.2f} | "
                + f"WR {sp.win_rate:.0%} | "
                + f"PnL ${sp.total_pnl:+,.0f} | {sp.status}"
            )
            breakdown_lines.append(line)
        fields.append(
            {
                "name": "Strategy Breakdown (30d)",
                "value": "\n".join(breakdown_lines),
                "inline": False,
            }
        )

    # Alpha Decay Warning (non-inline 1, 감지 시에만)
    if alpha_decay and strategy_health:
        fields.append(
            {
                "name": "Alpha Decay Warning",
                "value": (
                    f"Rolling Sharpe 3-period decline detected "
                    f"(current: {strategy_health.rolling_sharpe_30d:.2f})"
                ),
                "inline": False,
            }
        )

    # Open Position Detail (non-inline 1)
    if strategy_health and strategy_health.open_positions:
        pos_lines = [
            f"{pos.direction} {pos.symbol}: ${pos.unrealized_pnl:+,.2f}"
            for pos in strategy_health.open_positions
        ]
        fields.append(
            {
                "name": f"Position Detail ({len(strategy_health.open_positions)})",
                "value": "\n".join(pos_lines),
                "inline": False,
            }
        )

    # System Status (inline 3)
    if system_health is not None:
        from src.notification.health_formatters import format_uptime

        cb_label = "ACTIVE" if system_health.is_circuit_breaker_active else "OK"
        ws_ok = system_health.total_symbols - system_health.stale_symbol_count
        fields.extend(
            [
                {
                    "name": "Uptime",
                    "value": format_uptime(system_health.uptime_seconds),
                    "inline": True,
                },
                {"name": "CB Status", "value": cb_label, "inline": True},
                {
                    "name": "WS Status",
                    "value": f"{ws_ok}/{system_health.total_symbols} OK",
                    "inline": True,
                },
            ]
        )

    return {
        "title": "Daily Report",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def _build_strategy_info_field(data: StrategyInfoMixin) -> dict[str, Any]:
    """Section: Strategy Info embed field."""
    params_str = " | ".join(f"{k}: {v}" for k, v in data.strategy_params.items())
    return {
        "name": f"Strategy: {data.strategy_name}",
        "value": f"{params_str}\nTS: {data.trailing_stop_config} | TF: {data.timeframe}",
        "inline": False,
    }


def _build_indicator_fields(indicators: tuple[StrategyIndicatorItem, ...]) -> dict[str, Any]:
    """Section: Strategy Indicators embed field."""
    ind_lines: list[str] = []
    for i in indicators:
        st_str = f"ST {_fmt_price(i.supertrend_line)}" if i.supertrend_line else "ST —"
        adx_str = f"ADX {i.adx_value:.1f}" if i.adx_value is not None else "ADX —"
        ind_lines.append(f"**{i.symbol}** {st_str} | {adx_str} | {i.outlook}")
    return {
        "name": "Strategy Indicators",
        "value": "\n".join(ind_lines) if ind_lines else "—",
        "inline": False,
    }


def _build_system_health_fields(data: SystemHealthMixin) -> list[dict[str, Any]]:
    """Section: System Health embed fields."""
    from src.notification.health_formatters import format_uptime

    cb_label = "ACTIVE" if data.is_circuit_breaker_active else "OK"
    pf_str = f"{data.profit_factor:.2f}" if data.profit_factor != float("inf") else "INF"
    return [
        {"name": "Uptime", "value": format_uptime(data.uptime_seconds), "inline": True},
        {
            "name": "CB / WS",
            "value": f"{cb_label} | {data.ws_ok_count}/{data.ws_total_count}",
            "inline": True,
        },
        {
            "name": "Sharpe (30d)",
            "value": f"{data.rolling_sharpe_30d:.2f}",
            "inline": True,
        },
        {"name": "Win Rate", "value": f"{data.win_rate:.0%}", "inline": True},
        {"name": "Profit Factor", "value": pf_str, "inline": True},
        {
            "name": "Alpha Decay",
            "value": "DETECTED" if data.alpha_decay_detected else "OK",
            "inline": True,
        },
    ]


def format_spot_daily_report_embed(data: DailyReportData) -> dict[str, Any]:
    """Spot Daily Report -> 5-section Discord Embed.

    Args:
        data: DailyReportData (HealthDataCollector에서 수집)

    Returns:
        Discord Embed dict
    """
    color = _COLOR_RED if data.alpha_decay_detected else _COLOR_BLUE
    fields: list[dict[str, Any]] = []

    # Section 1: Strategy Info
    fields.append(_build_strategy_info_field(data))

    # Section 2: Portfolio Summary
    fields.extend(
        [
            {"name": "Equity", "value": f"${data.total_equity:,.0f}", "inline": True},
            {
                "name": "Cash",
                "value": f"${data.available_cash:,.0f} ({data.cash_pct:.1f}%)",
                "inline": True,
            },
            {"name": "Today PnL", "value": f"${data.today_pnl:+,.2f}", "inline": True},
            {
                "name": "Invested",
                "value": f"{data.invested_count}/{data.total_asset_count} assets",
                "inline": True,
            },
            {
                "name": "Cum. Return",
                "value": f"{data.cumulative_return_pct:+.2f}%",
                "inline": True,
            },
            {"name": "MDD", "value": f"{data.max_drawdown_pct:.1f}%", "inline": True},
        ]
    )

    # Section 3: Asset Dashboard
    if data.assets:
        lines: list[str] = []
        for a in data.assets:
            if a.stop_distance_pct is not None:
                stop_str = f"Stop {a.stop_distance_pct:.1f}%"
            else:
                stop_str = "—"
            if a.position_value > 0:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.change_24h_pct:+.1f}%) | "
                    f"${a.position_value:,.0f} | "
                    f"PnL ${a.day_pnl:+,.2f} | {stop_str}"
                )
            else:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.change_24h_pct:+.1f}%)"
                )
            lines.append(line)
        fields.append(
            {
                "name": f"Asset Dashboard ({len(data.assets)})",
                "value": "\n".join(lines),
                "inline": False,
            }
        )

    # Section 4: Strategy Indicators
    if data.indicators:
        fields.append(_build_indicator_fields(data.indicators))

    # Section 5: System Health
    from src.notification.health_formatters import format_uptime

    cb_label = "ACTIVE" if data.is_circuit_breaker_active else "OK"
    pf_str = f"{data.profit_factor:.2f}" if data.profit_factor != float("inf") else "INF"
    fields.extend(
        [
            {"name": "Uptime", "value": format_uptime(data.uptime_seconds), "inline": True},
            {
                "name": "CB / WS",
                "value": f"{cb_label} | {data.ws_ok_count}/{data.ws_total_count}",
                "inline": True,
            },
            {
                "name": "Sharpe (30d)",
                "value": f"{data.rolling_sharpe_30d:.2f}",
                "inline": True,
            },
            {"name": "Win Rate", "value": f"{data.win_rate:.0%}", "inline": True},
            {"name": "Profit Factor", "value": pf_str, "inline": True},
            {
                "name": "Alpha Decay",
                "value": "DETECTED" if data.alpha_decay_detected else "OK",
                "inline": True,
            },
        ]
    )

    return {
        "title": "Spot Daily Report",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_bar_close_report_embed(data: BarCloseReportData) -> dict[str, Any]:
    """12H Bar Close Report -> Discord Embed.

    Args:
        data: BarCloseReportData (HealthDataCollector에서 수집)

    Returns:
        Discord Embed dict
    """
    from src.notification.health_formatters import format_uptime

    color = _COLOR_RED if data.is_circuit_breaker_active else _COLOR_GREEN
    fields: list[dict[str, Any]] = []

    # Section 1: Signal Changes
    if data.signal_changes:
        sc_lines: list[str] = []
        for sc in data.signal_changes:
            pnl_str = f" (PnL ${sc.realized_pnl:+,.2f})" if sc.realized_pnl is not None else ""
            sc_lines.append(f"{sc.symbol}: {sc.prev_signal} -> {sc.new_signal}{pnl_str}")
        fields.append({"name": "Signal Changes", "value": "\n".join(sc_lines), "inline": False})
    else:
        fields.append({"name": "Signal Changes", "value": "No changes", "inline": False})

    # Section 2: Asset Dashboard (compact)
    if data.assets:
        lines: list[str] = []
        for a in data.assets:
            if a.position_value > 0:
                stop_str = (
                    f"| Stop {a.stop_distance_pct:.1f}%" if a.stop_distance_pct is not None else ""
                )
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.change_24h_pct:+.1f}%) | "
                    f"${a.position_value:,.0f} {stop_str}"
                )
            else:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.change_24h_pct:+.1f}%)"
                )
            lines.append(line)
        fields.append(
            {
                "name": f"Assets ({len(data.assets)})",
                "value": "\n".join(lines),
                "inline": False,
            }
        )

    # Section 3: Portfolio Snapshot
    fields.extend(
        [
            {"name": "Equity", "value": f"${data.total_equity:,.0f}", "inline": True},
            {"name": "Cash", "value": f"${data.available_cash:,.0f}", "inline": True},
            {"name": "Deployed", "value": f"${data.capital_deployed:,.0f}", "inline": True},
            {"name": "Today PnL", "value": f"${data.today_pnl:+,.2f}", "inline": True},
            {"name": "Drawdown", "value": f"{data.drawdown_pct:.1%}", "inline": True},
            {"name": "Utilization", "value": f"{data.capital_utilization:.1%}", "inline": True},
            {
                "name": "Invested",
                "value": f"{data.invested_count}/{data.total_asset_count}",
                "inline": True,
            },
        ]
    )

    # Section 4: System Status (one line)
    cb_label = "ACTIVE" if data.is_circuit_breaker_active else "OK"
    status_str = (
        f"CB {cb_label} | WS {data.ws_ok_count}/{data.ws_total_count} | "
        f"Uptime {format_uptime(data.uptime_seconds)}"
    )
    fields.append({"name": "System", "value": status_str, "inline": False})

    return {
        "title": f"12H Bar Close ({data.bar_time_utc} UTC)",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_spot_weekly_report_embed(data: WeeklyReportData) -> dict[str, Any]:
    """Spot Weekly Report -> 6-section Discord Embed.

    Args:
        data: WeeklyReportData (HealthDataCollector에서 수집)

    Returns:
        Discord Embed dict
    """
    color = _COLOR_RED if data.alpha_decay_detected else _COLOR_BLUE
    fields: list[dict[str, Any]] = []

    # Section 1: Strategy Info
    fields.append(_build_strategy_info_field(data))

    # Section 2: Weekly Portfolio Summary
    fields.extend(
        [
            {"name": "Equity", "value": f"${data.total_equity:,.0f}", "inline": True},
            {
                "name": "Cash",
                "value": f"${data.available_cash:,.0f} ({data.cash_pct:.1f}%)",
                "inline": True,
            },
            {"name": "Week PnL", "value": f"${data.week_pnl:+,.2f}", "inline": True},
            {
                "name": "Week Trades",
                "value": str(data.week_trades),
                "inline": True,
            },
            {
                "name": "Invested",
                "value": f"{data.invested_count}/{data.total_asset_count} assets",
                "inline": True,
            },
            {
                "name": "Cum. Return",
                "value": f"{data.cumulative_return_pct:+.2f}%",
                "inline": True,
            },
            {"name": "MDD", "value": f"{data.max_drawdown_pct:.1f}%", "inline": True},
        ]
    )

    # Section 3: Asset Weekly Performance
    if data.assets:
        lines: list[str] = []
        for a in data.assets:
            if a.week_trades > 0:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.week_change_pct:+.1f}%) | "
                    f"PnL ${a.week_pnl:+,.2f} | {a.week_trades} trades"
                )
            else:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.week_change_pct:+.1f}%)"
                )
            lines.append(line)
        fields.append(
            {
                "name": f"Asset Weekly Performance ({len(data.assets)})",
                "value": "\n".join(lines),
                "inline": False,
            }
        )

    # Section 4: Weekly Trade Summary
    pf_str = f"{data.week_profit_factor:.2f}" if data.week_profit_factor != float("inf") else "INF"
    fields.extend(
        [
            {
                "name": "Best Trade",
                "value": f"{data.best_trade_symbol} ${data.best_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Worst Trade",
                "value": f"{data.worst_trade_symbol} ${data.worst_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Week WR / PF",
                "value": f"{data.week_win_rate:.0%} / {pf_str}",
                "inline": True,
            },
        ]
    )

    # Section 5: Strategy Indicators
    if data.indicators:
        fields.append(_build_indicator_fields(data.indicators))

    # Section 6: System Health
    fields.extend(_build_system_health_fields(data))

    return {
        "title": "Spot Weekly Report",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_spot_monthly_report_embed(data: MonthlyReportData) -> dict[str, Any]:
    """Spot Monthly Report -> 8-section Discord Embed.

    Args:
        data: MonthlyReportData (HealthDataCollector에서 수집)

    Returns:
        Discord Embed dict
    """
    color = _COLOR_RED if data.alpha_decay_detected else _COLOR_BLUE
    fields: list[dict[str, Any]] = []

    # Section 1: Strategy Info
    fields.append(_build_strategy_info_field(data))

    # Section 2: Monthly Portfolio Summary
    fields.extend(
        [
            {"name": "Equity", "value": f"${data.total_equity:,.0f}", "inline": True},
            {
                "name": "Cash",
                "value": f"${data.available_cash:,.0f} ({data.cash_pct:.1f}%)",
                "inline": True,
            },
            {
                "name": "Month PnL",
                "value": f"${data.month_pnl:+,.2f} ({data.month_return_pct:+.1f}%)",
                "inline": True,
            },
            {
                "name": "Month Trades",
                "value": str(data.month_trades),
                "inline": True,
            },
            {
                "name": "Invested",
                "value": f"{data.invested_count}/{data.total_asset_count} assets",
                "inline": True,
            },
            {
                "name": "Cum. Return",
                "value": f"{data.cumulative_return_pct:+.2f}%",
                "inline": True,
            },
            {"name": "MDD", "value": f"{data.max_drawdown_pct:.1f}%", "inline": True},
        ]
    )

    # Section 3: Asset Monthly Performance
    if data.assets:
        lines: list[str] = []
        for a in data.assets:
            if a.month_trades > 0:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.month_change_pct:+.1f}%) | "
                    f"PnL ${a.month_pnl:+,.2f} | {a.month_trades} trades"
                )
            else:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.month_change_pct:+.1f}%)"
                )
            lines.append(line)
        fields.append(
            {
                "name": f"Asset Monthly Performance ({len(data.assets)})",
                "value": "\n".join(lines),
                "inline": False,
            }
        )

    # Section 4: Monthly Trade Summary
    pf_str = (
        f"{data.month_profit_factor:.2f}" if data.month_profit_factor != float("inf") else "INF"
    )
    fields.extend(
        [
            {
                "name": "Best Trade",
                "value": f"{data.best_trade_symbol} ${data.best_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Worst Trade",
                "value": f"{data.worst_trade_symbol} ${data.worst_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Month WR / PF",
                "value": f"{data.month_win_rate:.0%} / {pf_str}",
                "inline": True,
            },
            {
                "name": "Avg Trade PnL",
                "value": f"${data.avg_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Total Fees",
                "value": f"${data.total_fees:,.2f}",
                "inline": True,
            },
        ]
    )

    # Section 5: Performance Trend
    if data.performance_trend:
        trend_lines = [
            (
                f"**{t.year_month}**: ${t.pnl:+,.0f} ({t.return_pct:+.1f}%) | "
                f"{t.trades} trades | Sharpe {t.sharpe:.2f}"
            )
            for t in data.performance_trend
        ]
        fields.append(
            {
                "name": "Performance Trend",
                "value": "\n".join(trend_lines),
                "inline": False,
            }
        )

    # Section 6: Strategy Indicators
    if data.indicators:
        fields.append(_build_indicator_fields(data.indicators))

    # Section 7: System Health
    fields.extend(_build_system_health_fields(data))

    # Section 8: Risk Summary
    fields.extend(
        [
            {
                "name": "Month Max DD",
                "value": f"{data.month_max_drawdown_pct:.2f}%",
                "inline": True,
            },
            {
                "name": "Longest Losing Streak",
                "value": str(data.longest_losing_streak),
                "inline": True,
            },
        ]
    )

    return {
        "title": "Spot Monthly Report",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_spot_quarterly_report_embed(data: QuarterlyReportData) -> dict[str, Any]:
    """Spot Quarterly Report -> 8-section Discord Embed."""
    color = _COLOR_RED if data.alpha_decay_detected else _COLOR_BLUE
    fields: list[dict[str, Any]] = []

    # Section 1: Strategy Info
    fields.append(_build_strategy_info_field(data))

    # Section 2: Quarterly Portfolio Summary
    fields.extend(
        [
            {"name": "Equity", "value": f"${data.total_equity:,.0f}", "inline": True},
            {
                "name": "Cash",
                "value": f"${data.available_cash:,.0f} ({data.cash_pct:.1f}%)",
                "inline": True,
            },
            {
                "name": "Quarter PnL",
                "value": f"${data.quarter_pnl:+,.2f} ({data.quarter_return_pct:+.1f}%)",
                "inline": True,
            },
            {
                "name": "Quarter Trades",
                "value": str(data.quarter_trades),
                "inline": True,
            },
            {
                "name": "Invested",
                "value": f"{data.invested_count}/{data.total_asset_count} assets",
                "inline": True,
            },
            {
                "name": "Cum. Return",
                "value": f"{data.cumulative_return_pct:+.2f}%",
                "inline": True,
            },
            {"name": "MDD", "value": f"{data.max_drawdown_pct:.1f}%", "inline": True},
        ]
    )

    # Section 3: Asset Quarterly Performance
    if data.assets:
        lines: list[str] = []
        for a in data.assets:
            if a.quarter_trades > 0:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.quarter_change_pct:+.1f}%) | "
                    f"PnL ${a.quarter_pnl:+,.2f} | {a.quarter_trades} trades"
                )
            else:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.quarter_change_pct:+.1f}%)"
                )
            lines.append(line)
        fields.append(
            {
                "name": f"Asset Quarterly Performance ({len(data.assets)})",
                "value": "\n".join(lines),
                "inline": False,
            }
        )

    # Section 4: Quarterly Trade Summary
    pf_str = (
        f"{data.quarter_profit_factor:.2f}" if data.quarter_profit_factor != float("inf") else "INF"
    )
    fields.extend(
        [
            {
                "name": "Best Trade",
                "value": f"{data.best_trade_symbol} ${data.best_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Worst Trade",
                "value": f"{data.worst_trade_symbol} ${data.worst_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Quarter WR / PF",
                "value": f"{data.quarter_win_rate:.0%} / {pf_str}",
                "inline": True,
            },
            {
                "name": "Avg Trade PnL",
                "value": f"${data.avg_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Total Fees",
                "value": f"${data.total_fees:,.2f}",
                "inline": True,
            },
        ]
    )

    # Section 5: Monthly Performance Trend
    if data.performance_trend:
        trend_lines = [
            (
                f"**{t.year_month}**: ${t.pnl:+,.0f} ({t.return_pct:+.1f}%) | "
                f"{t.trades} trades | Sharpe {t.sharpe:.2f}"
            )
            for t in data.performance_trend
        ]
        fields.append(
            {
                "name": "Monthly Trend (3M)",
                "value": "\n".join(trend_lines),
                "inline": False,
            }
        )

    # Section 6: Strategy Indicators
    if data.indicators:
        fields.append(_build_indicator_fields(data.indicators))

    # Section 7: System Health
    fields.extend(_build_system_health_fields(data))

    # Section 8: Risk Summary
    fields.extend(
        [
            {
                "name": "Quarter Max DD",
                "value": f"{data.quarter_max_drawdown_pct:.2f}%",
                "inline": True,
            },
            {
                "name": "Longest Losing Streak",
                "value": str(data.longest_losing_streak),
                "inline": True,
            },
        ]
    )

    return {
        "title": "Spot Quarterly Report",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_spot_yearly_report_embed(data: YearlyReportData) -> dict[str, Any]:
    """Spot Yearly Report -> 9-section Discord Embed."""
    color = _COLOR_RED if data.alpha_decay_detected else _COLOR_BLUE
    fields: list[dict[str, Any]] = []

    # Section 1: Strategy Info
    fields.append(_build_strategy_info_field(data))

    # Section 2: Yearly Portfolio Summary
    fields.extend(
        [
            {"name": "Equity", "value": f"${data.total_equity:,.0f}", "inline": True},
            {
                "name": "Cash",
                "value": f"${data.available_cash:,.0f} ({data.cash_pct:.1f}%)",
                "inline": True,
            },
            {
                "name": "Year PnL",
                "value": f"${data.year_pnl:+,.2f} ({data.year_return_pct:+.1f}%)",
                "inline": True,
            },
            {
                "name": "Year Trades",
                "value": str(data.year_trades),
                "inline": True,
            },
            {
                "name": "Invested",
                "value": f"{data.invested_count}/{data.total_asset_count} assets",
                "inline": True,
            },
            {
                "name": "Cum. Return",
                "value": f"{data.cumulative_return_pct:+.2f}%",
                "inline": True,
            },
            {"name": "MDD", "value": f"{data.max_drawdown_pct:.1f}%", "inline": True},
        ]
    )

    # Section 3: Asset Yearly Performance
    if data.assets:
        lines: list[str] = []
        for a in data.assets:
            if a.year_trades > 0:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.year_change_pct:+.1f}%) | "
                    f"PnL ${a.year_pnl:+,.2f} | {a.year_trades} trades"
                )
            else:
                line = (
                    f"**{a.symbol}** {a.signal} | "
                    f"{_fmt_price(a.current_price)} ({a.year_change_pct:+.1f}%)"
                )
            lines.append(line)
        fields.append(
            {
                "name": f"Asset Yearly Performance ({len(data.assets)})",
                "value": "\n".join(lines),
                "inline": False,
            }
        )

    # Section 4: Yearly Trade Summary
    pf_str = f"{data.year_profit_factor:.2f}" if data.year_profit_factor != float("inf") else "INF"
    fields.extend(
        [
            {
                "name": "Best Trade",
                "value": f"{data.best_trade_symbol} ${data.best_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Worst Trade",
                "value": f"{data.worst_trade_symbol} ${data.worst_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Year WR / PF",
                "value": f"{data.year_win_rate:.0%} / {pf_str}",
                "inline": True,
            },
            {
                "name": "Avg Trade PnL",
                "value": f"${data.avg_trade_pnl:+,.2f}",
                "inline": True,
            },
            {
                "name": "Total Fees",
                "value": f"${data.total_fees:,.2f}",
                "inline": True,
            },
        ]
    )

    # Section 5: Quarterly Performance Trend
    if data.quarterly_trend:
        q_lines = [
            (
                f"**{t.year_quarter}**: ${t.pnl:+,.0f} ({t.return_pct:+.1f}%) | "
                f"{t.trades} trades | Sharpe {t.sharpe:.2f}"
            )
            for t in data.quarterly_trend
        ]
        fields.append(
            {
                "name": "Quarterly Trend",
                "value": "\n".join(q_lines),
                "inline": False,
            }
        )

    # Section 6: Monthly Performance Trend
    if data.monthly_trend:
        m_lines = [
            (
                f"**{t.year_month}**: ${t.pnl:+,.0f} ({t.return_pct:+.1f}%) | "
                f"{t.trades} trades | Sharpe {t.sharpe:.2f}"
            )
            for t in data.monthly_trend
        ]
        fields.append(
            {
                "name": "Monthly Trend (12M)",
                "value": "\n".join(m_lines),
                "inline": False,
            }
        )

    # Section 7: Strategy Indicators
    if data.indicators:
        fields.append(_build_indicator_fields(data.indicators))

    # Section 8: System Health
    fields.extend(_build_system_health_fields(data))

    # Section 9: Risk Summary
    fields.extend(
        [
            {
                "name": "Year Max DD",
                "value": f"{data.year_max_drawdown_pct:.2f}%",
                "inline": True,
            },
            {
                "name": "Longest Losing Streak",
                "value": str(data.longest_losing_streak),
                "inline": True,
            },
        ]
    )

    return {
        "title": "Spot Yearly Report",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }
