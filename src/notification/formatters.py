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
        MarketRegimeReport,
        StrategyHealthSnapshot,
        SystemHealthSnapshot,
    )

# Discord Embed 색상 코드 (decimal)
_COLOR_GREEN = 0x57F287
_COLOR_RED = 0xED4245
_COLOR_BLUE = 0x3498DB
_COLOR_YELLOW = 0xFFFF00
_COLOR_ORANGE = 0xE67E22

_FOOTER_TEXT = "MC-Coin-Bot"


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
                "name": "Margin Used",
                "value": f"${event.total_margin_used:,.2f}",
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


def format_enhanced_daily_report_embed(
    metrics: PerformanceMetrics,
    open_positions: int,
    total_equity: float,
    trades_today: list[TradeRecord],
    system_health: SystemHealthSnapshot | None = None,
    strategy_health: StrategyHealthSnapshot | None = None,
    regime_report: MarketRegimeReport | None = None,
) -> dict[str, Any]:
    """Enhanced Daily Report -> Discord Embed dict.

    기본 Daily Report에 시스템/전략/마켓 건강 데이터를 통합합니다.
    health 데이터가 None이면 기본 Daily Report와 동일한 결과를 반환합니다.

    Args:
        metrics: 현재 성과 지표
        open_positions: 오픈 포지션 수
        total_equity: 현재 equity
        trades_today: 오늘 거래 목록
        system_health: 시스템 건강 스냅샷 (None이면 생략)
        strategy_health: 전략 건강 스냅샷 (None이면 생략)
        regime_report: 마켓 regime 리포트 (None이면 생략)

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

    # Market Regime (inline 2)
    if regime_report is not None:
        fields.append(
            {
                "name": "Market Regime",
                "value": f"{regime_report.regime_label} ({regime_report.regime_score:+.2f})",
                "inline": True,
            }
        )
        # Top funding rates
        if regime_report.symbols:
            top_fr = sorted(regime_report.symbols, key=lambda s: abs(s.funding_rate), reverse=True)
            fr_lines = [f"{s.symbol}: {s.funding_rate * 100:+.3f}%" for s in top_fr[:3]]
            fields.append(
                {
                    "name": "Top Funding Rates",
                    "value": "\n".join(fr_lines),
                    "inline": True,
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


def format_surveillance_scan_embed(
    scan_result: Any,
    pod_additions: dict[str, list[str]],
) -> dict[str, Any]:
    """Surveillance 스캔 결과 Discord Embed.

    Args:
        scan_result: ScanResult 인스턴스
        pod_additions: Pod별 추가된 심볼 {pod_id: [symbol, ...]}

    Returns:
        Discord Embed dict
    """
    has_dropped = len(scan_result.dropped) > 0
    color = _COLOR_YELLOW if has_dropped else _COLOR_GREEN

    fields: list[dict[str, Any]] = [
        {
            "name": "Universe",
            "value": f"{len(scan_result.qualified_symbols)} assets",
            "inline": True,
        },
        {
            "name": "Scan Duration",
            "value": f"{scan_result.scan_duration_seconds:.1f}s",
            "inline": True,
        },
        {
            "name": "Total Scanned",
            "value": str(scan_result.total_scanned),
            "inline": True,
        },
    ]

    max_display = 10
    if scan_result.added:
        added_str = ", ".join(scan_result.added[:max_display])
        if len(scan_result.added) > max_display:
            added_str += f" (+{len(scan_result.added) - max_display} more)"
        fields.append({"name": "Added", "value": added_str, "inline": False})

    if scan_result.dropped:
        dropped_str = ", ".join(scan_result.dropped[:max_display])
        if len(scan_result.dropped) > max_display:
            dropped_str += f" (+{len(scan_result.dropped) - max_display} more)"
        fields.append({"name": "Dropped", "value": dropped_str, "inline": False})

    if pod_additions:
        pod_lines = [f"{pid}: +{len(syms)}" for pid, syms in pod_additions.items()]
        fields.append(
            {
                "name": "Pod Assignments",
                "value": ", ".join(pod_lines),
                "inline": False,
            }
        )

    return {
        "title": "Market Surveillance Scan",
        "color": color,
        "fields": fields,
        "timestamp": scan_result.timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }
