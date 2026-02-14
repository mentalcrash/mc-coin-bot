"""Bot Lifecycle Discord Embed 포매터.

봇 시작/종료/비정상종료 시 Discord ALERTS 채널에 전송할 embed를 생성합니다.

Rules Applied:
    - #10 Python Standards: Pure functions, type hints
    - #22 Notification Standards: Rich Embeds, color segmentation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# Discord Embed 색상 코드 (decimal) — formatters.py와 동일
_COLOR_GREEN = 0x57F287
_COLOR_RED = 0xED4245
_COLOR_YELLOW = 0xFFFF00

_FOOTER_TEXT = "MC-Coin-Bot"
_MAX_ERROR_MSG_LEN = 200


def _format_uptime(seconds: float) -> str:
    """Uptime을 사람이 읽기 좋은 형식으로 변환."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    parts: list[str] = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


def format_startup_embed(
    *,
    mode: str,
    strategy_name: str,
    symbols: list[str],
    capital: float,
    timeframe: str,
) -> dict[str, Any]:
    """봇 시작 → GREEN embed.

    Args:
        mode: 실행 모드 (paper/shadow/live)
        strategy_name: 전략 이름
        symbols: 거래 심볼 리스트
        capital: 초기 자본
        timeframe: 타임프레임

    Returns:
        Discord Embed dict
    """
    symbols_str = ", ".join(symbols) if symbols else "N/A"

    return {
        "title": "Bot Started",
        "color": _COLOR_GREEN,
        "fields": [
            {"name": "Mode", "value": mode.upper(), "inline": True},
            {"name": "Strategy", "value": strategy_name, "inline": True},
            {"name": "Timeframe", "value": timeframe, "inline": True},
            {"name": "Capital", "value": f"${capital:,.0f}", "inline": True},
            {"name": "Symbols", "value": symbols_str, "inline": False},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_shutdown_embed(
    *,
    reason: str,
    uptime_seconds: float,
    final_equity: float,
    today_pnl: float,
    open_positions: int,
) -> dict[str, Any]:
    """봇 정상 종료 → YELLOW embed.

    Args:
        reason: 종료 사유 (예: "SIGTERM (graceful)")
        uptime_seconds: 가동 시간 (초)
        final_equity: 종료 시 equity
        today_pnl: 금일 실현 PnL
        open_positions: 오픈 포지션 수

    Returns:
        Discord Embed dict
    """
    return {
        "title": "Bot Stopped",
        "color": _COLOR_YELLOW,
        "fields": [
            {"name": "Reason", "value": reason, "inline": True},
            {"name": "Uptime", "value": _format_uptime(uptime_seconds), "inline": True},
            {"name": "Final Equity", "value": f"${final_equity:,.0f}", "inline": True},
            {"name": "Today PnL", "value": f"${today_pnl:+,.0f}", "inline": True},
            {"name": "Open Positions", "value": str(open_positions), "inline": True},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_crash_embed(
    *,
    error_type: str,
    error_message: str,
    uptime_seconds: float,
) -> dict[str, Any]:
    """봇 비정상 종료 → RED embed (CRITICAL).

    Args:
        error_type: 예외 타입 이름
        error_message: 예외 메시지 (200자 제한)
        uptime_seconds: 가동 시간 (초)

    Returns:
        Discord Embed dict
    """
    # Discord embed value 길이 제한
    truncated_msg = (
        error_message[:_MAX_ERROR_MSG_LEN] + "..."
        if len(error_message) > _MAX_ERROR_MSG_LEN
        else error_message
    )

    return {
        "title": "Bot CRASHED",
        "color": _COLOR_RED,
        "fields": [
            {"name": "Error Type", "value": error_type, "inline": True},
            {"name": "Uptime", "value": _format_uptime(uptime_seconds), "inline": True},
            {"name": "Error", "value": truncated_msg, "inline": False},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }
