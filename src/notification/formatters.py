"""Event -> Discord Embed 변환 함수.

각 EDA 이벤트 타입을 Discord Embed dict로 포맷팅합니다.
DiscordNotifier의 기존 embed 스타일(색상, 필드 레이아웃)을 따릅니다.

Rules Applied:
    - #10 Python Standards: Pure functions, type hints
    - #22 Notification Standards: Rich Embeds, color segmentation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.events import (
        BalanceUpdateEvent,
        CircuitBreakerEvent,
        FillEvent,
        PositionUpdateEvent,
        RiskAlertEvent,
    )

# Discord Embed 색상 코드 (decimal)
_COLOR_GREEN = 0x57F287
_COLOR_RED = 0xED4245
_COLOR_BLUE = 0x3498DB
_COLOR_YELLOW = 0xFFFF00
_COLOR_ORANGE = 0xE67E22

_FOOTER_TEXT = "MC-Coin-Bot"


def format_fill_embed(fill: FillEvent) -> dict[str, Any]:
    """FillEvent -> Discord Embed dict.

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
