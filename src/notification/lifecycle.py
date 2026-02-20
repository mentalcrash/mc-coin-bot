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


def _format_pod_table(pod_summaries: list[dict[str, object]]) -> str:
    """Pod 요약 리스트를 monospace 테이블 문자열로 변환.

    orchestrator_formatters.py의 Pod 테이블 패턴 재사용.
    """
    lines: list[str] = [
        "`Pod              State       Alloc`",
        "`──────────────────────────────────`",
    ]
    for summary in pod_summaries:
        pid = str(summary.get("pod_id", ""))
        state = str(summary.get("state", ""))
        raw_frac = summary.get("capital_fraction", 0.0)
        frac = float(raw_frac) if isinstance(raw_frac, (int, float)) else 0.0
        lines.append(f"`{pid:<16} {state:<11} {frac:>5.1%}`")
    return "\n".join(lines)


def format_startup_embed(
    *,
    mode: str,
    strategy_name: str,
    symbols: list[str],
    capital: float,
    timeframe: str,
    pod_summaries: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """봇 시작 → GREEN embed.

    Args:
        mode: 실행 모드 (paper/shadow/live)
        strategy_name: 전략 이름
        symbols: 거래 심볼 리스트
        capital: 초기 자본
        timeframe: 타임프레임
        pod_summaries: Orchestrator pod 요약 (None이면 Pods 필드 미포함)

    Returns:
        Discord Embed dict
    """
    symbols_str = ", ".join(symbols) if symbols else "N/A"

    fields: list[dict[str, Any]] = [
        {"name": "Mode", "value": mode.upper(), "inline": True},
        {"name": "Strategy", "value": strategy_name, "inline": True},
        {"name": "Timeframe", "value": timeframe, "inline": True},
        {"name": "Capital", "value": f"${capital:,.0f}", "inline": True},
        {"name": "Symbols", "value": symbols_str, "inline": False},
    ]

    if pod_summaries is not None:
        fields.append({"name": "Pods", "value": _format_pod_table(pod_summaries), "inline": False})

    return {
        "title": "MC Coin Bot Started",
        "color": _COLOR_GREEN,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_shutdown_embed(
    *,
    reason: str,
    uptime_seconds: float,
    final_equity: float,
    initial_capital: float,
    realized_pnl: float,
    unrealized_pnl: float,
    open_positions: int,
    pod_summaries: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """봇 정상 종료 → YELLOW embed.

    Args:
        reason: 종료 사유 (예: "SIGTERM (graceful)")
        uptime_seconds: 가동 시간 (초)
        final_equity: 종료 시 equity
        initial_capital: 초기 자본 (PnL 퍼센트 계산용)
        realized_pnl: 금일 실현 PnL
        unrealized_pnl: 미실현 PnL
        open_positions: 오픈 포지션 수
        pod_summaries: Orchestrator pod 요약 (None이면 Pods 필드 미포함)

    Returns:
        Discord Embed dict
    """
    total_pnl = realized_pnl + unrealized_pnl
    pnl_pct = (total_pnl / initial_capital * 100) if initial_capital != 0 else 0.0

    fields: list[dict[str, Any]] = [
        {"name": "Reason", "value": reason, "inline": True},
        {"name": "Uptime", "value": _format_uptime(uptime_seconds), "inline": True},
        {"name": "Final Equity", "value": f"${final_equity:,.0f}", "inline": True},
        {
            "name": "Today PnL",
            "value": f"${total_pnl:+,.2f} ({pnl_pct:+.2f}%)",
            "inline": True,
        },
        {"name": "Realized", "value": f"${realized_pnl:+,.2f}", "inline": True},
        {"name": "Unrealized", "value": f"${unrealized_pnl:+,.2f}", "inline": True},
        {"name": "Open Positions", "value": str(open_positions), "inline": True},
    ]

    if pod_summaries is not None:
        fields.append({"name": "Pods", "value": _format_pod_table(pod_summaries), "inline": False})

    return {
        "title": "MC Coin Bot Stopped",
        "color": _COLOR_YELLOW,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_crash_embed(
    *,
    error_type: str,
    error_message: str,
    uptime_seconds: float,
    final_equity: float | None = None,
    open_positions: int | None = None,
    unrealized_pnl: float | None = None,
) -> dict[str, Any]:
    """봇 비정상 종료 → RED embed (CRITICAL).

    Args:
        error_type: 예외 타입 이름
        error_message: 예외 메시지 (200자 제한)
        uptime_seconds: 가동 시간 (초)
        final_equity: 종료 시 equity (None이면 필드 미포함)
        open_positions: 오픈 포지션 수 (None이면 필드 미포함)
        unrealized_pnl: 미실현 PnL (None이면 필드 미포함)

    Returns:
        Discord Embed dict
    """
    # Discord embed value 길이 제한
    truncated_msg = (
        error_message[:_MAX_ERROR_MSG_LEN] + "..."
        if len(error_message) > _MAX_ERROR_MSG_LEN
        else error_message
    )

    fields: list[dict[str, Any]] = [
        {"name": "Error Type", "value": error_type, "inline": True},
        {"name": "Uptime", "value": _format_uptime(uptime_seconds), "inline": True},
        {"name": "Error", "value": truncated_msg, "inline": False},
    ]

    if final_equity is not None:
        fields.append({"name": "Final Equity", "value": f"${final_equity:,.0f}", "inline": True})
    if open_positions is not None:
        fields.append({"name": "Open Positions", "value": str(open_positions), "inline": True})
    if unrealized_pnl is not None:
        fields.append(
            {"name": "Unrealized PnL", "value": f"${unrealized_pnl:+,.2f}", "inline": True}
        )

    return {
        "title": "MC Coin Bot CRASHED",
        "color": _COLOR_RED,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }
