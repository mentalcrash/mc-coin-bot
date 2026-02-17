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
