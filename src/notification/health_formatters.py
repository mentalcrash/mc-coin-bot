"""Health Check → Discord Embed 변환 함수.

HealthCheckScheduler에서 수집한 스냅샷을 Discord Embed dict로 변환합니다.
formatters.py 패턴을 따르며, 순수 함수로 구현합니다.

Rules Applied:
    - #10 Python Standards: Pure functions, type hints
    - #22 Notification Standards: Rich Embeds, color segmentation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.notification.health_models import (
        MarketRegimeReport,
        StrategyHealthSnapshot,
        SystemHealthSnapshot,
    )

# Discord Embed 색상 코드 (decimal) — formatters.py와 동일
_COLOR_GREEN = 0x57F287
_COLOR_RED = 0xED4245
_COLOR_BLUE = 0x3498DB
_COLOR_YELLOW = 0xFFFF00

_FOOTER_TEXT = "MC-Coin-Bot"

# Heartbeat 임계값
_DD_YELLOW_THRESHOLD = 0.05  # 5%
_DD_RED_THRESHOLD = 0.08  # 8%
_QUEUE_DEPTH_YELLOW = 50
_SAFETY_STOP_FAILURE_THRESHOLD = 5

# Regime color 임계값 (regime_score.py 라벨과 동일)
_REGIME_EXTREME_THRESHOLD = 0.5
_REGIME_BULLISH_THRESHOLD = 0.2
_REGIME_BEARISH_THRESHOLD = -0.2


def _heartbeat_color(snapshot: SystemHealthSnapshot) -> int:
    """Heartbeat embed 색상 결정.

    🟢 GREEN: DD < 5%, stale 0개, CB 비활성, queue 정상
    🟡 YELLOW: DD 5~8%, or stale > 0, or queue depth > 50
    🔴 RED: DD > 8%, or CB 활성, or 전체 심볼 stale, or queue degraded
    """
    all_stale = snapshot.total_symbols > 0 and snapshot.stale_symbol_count >= snapshot.total_symbols

    # RED 조건
    has_red = (
        snapshot.is_circuit_breaker_active
        or snapshot.current_drawdown > _DD_RED_THRESHOLD
        or all_stale
        or snapshot.is_notification_degraded
        or snapshot.safety_stop_failures >= _SAFETY_STOP_FAILURE_THRESHOLD
    )
    if has_red:
        return _COLOR_RED

    # YELLOW 조건
    onchain_stale = (
        snapshot.onchain_sources_total > 0
        and snapshot.onchain_sources_ok < snapshot.onchain_sources_total
    )
    has_yellow = (
        snapshot.current_drawdown > _DD_YELLOW_THRESHOLD
        or snapshot.stale_symbol_count > 0
        or snapshot.max_queue_depth > _QUEUE_DEPTH_YELLOW
        or onchain_stale
    )
    if has_yellow:
        return _COLOR_YELLOW

    return _COLOR_GREEN


def format_uptime(seconds: float) -> str:
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


def format_heartbeat_embed(snapshot: SystemHealthSnapshot) -> dict[str, Any]:
    """SystemHealthSnapshot → Discord Embed dict (Tier 1).

    Args:
        snapshot: 시스템 건강 상태 스냅샷

    Returns:
        Discord Embed dict
    """
    color = _heartbeat_color(snapshot)
    status = "OK" if color == _COLOR_GREEN else ("WARN" if color == _COLOR_YELLOW else "ALERT")

    ws_ok = snapshot.total_symbols - snapshot.stale_symbol_count
    ws_label = f"{ws_ok}/{snapshot.total_symbols} OK"
    cb_label = "ACTIVE" if snapshot.is_circuit_breaker_active else "OK"
    dd_pct = snapshot.current_drawdown * 100

    fields: list[dict[str, Any]] = [
        {"name": "Uptime", "value": format_uptime(snapshot.uptime_seconds), "inline": True},
        {"name": "Equity", "value": f"${snapshot.total_equity:,.0f}", "inline": True},
        {"name": "Drawdown", "value": f"-{dd_pct:.1f}%", "inline": True},
        {"name": "WS Status", "value": ws_label, "inline": True},
        {"name": "Positions", "value": f"{snapshot.open_position_count}", "inline": True},
        {"name": "Leverage", "value": f"{snapshot.aggregate_leverage:.2f}x", "inline": True},
        {
            "name": "Today PnL",
            "value": f"${snapshot.today_pnl:+,.0f} ({snapshot.today_trades})",
            "inline": True,
        },
        {"name": "Queue Depth", "value": str(snapshot.max_queue_depth), "inline": True},
        {"name": "CB Status", "value": cb_label, "inline": True},
        {
            "name": "Safety Stops",
            "value": f"{snapshot.safety_stop_count} active"
            + (
                f" ({snapshot.safety_stop_failures} failures)"
                if snapshot.safety_stop_failures
                else ""
            ),
            "inline": True,
        },
    ]

    # On-chain 필드 (설정되어 있을 때만)
    if snapshot.onchain_sources_total > 0:
        ok = snapshot.onchain_sources_ok
        total = snapshot.onchain_sources_total
        label = f"{ok}/{total} OK ({snapshot.onchain_cache_columns} cols)"
        fields.append({"name": "On-chain", "value": label, "inline": True})

    return {
        "title": f"System Heartbeat — {status}",
        "color": color,
        "fields": fields,
        "timestamp": snapshot.timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def _regime_color(score: float) -> int:
    """Regime score에 따른 embed 색상."""
    if score > _REGIME_EXTREME_THRESHOLD or score < -_REGIME_EXTREME_THRESHOLD:
        return _COLOR_RED
    if score > _REGIME_BULLISH_THRESHOLD:
        return _COLOR_GREEN
    if score > _REGIME_BEARISH_THRESHOLD:
        return _COLOR_BLUE
    return _COLOR_YELLOW


def format_regime_embed(report: MarketRegimeReport) -> dict[str, Any]:
    """MarketRegimeReport → Discord Embed dict (Tier 2).

    Args:
        report: 마켓 regime 리포트

    Returns:
        Discord Embed dict
    """
    color = _regime_color(report.regime_score)

    # 심볼별 요약 라인
    lines: list[str] = []
    for sym in report.symbols:
        fr_pct = sym.funding_rate * 100
        ann_pct = sym.funding_rate_annualized
        line = (
            f"**{sym.symbol}**  ${sym.price:,.0f}\n"
            f"  FR {fr_pct:+.3f}% (ann {ann_pct:.1f}%) | "
            f"LS {sym.ls_ratio:.2f} | Taker {sym.taker_ratio:.2f}"
        )
        lines.append(line)

    description = "\n\n".join(lines) if lines else "No data available"

    return {
        "title": f"Market Regime — {report.regime_label} ({report.regime_score:+.2f})",
        "color": color,
        "description": description,
        "timestamp": report.timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_strategy_health_embed(snapshot: StrategyHealthSnapshot) -> dict[str, Any]:
    """StrategyHealthSnapshot → Discord Embed dict (Tier 3).

    Args:
        snapshot: 전략 건강 상태 스냅샷

    Returns:
        Discord Embed dict
    """
    # Alpha decay 경고 시 RED, 아니면 BLUE
    color = _COLOR_RED if snapshot.alpha_decay_detected else _COLOR_BLUE

    sharpe_arrow = ""
    if snapshot.alpha_decay_detected:
        sharpe_arrow = " (DECAY)"

    fields: list[dict[str, Any]] = [
        {
            "name": "Rolling Sharpe (30d)",
            "value": f"{snapshot.rolling_sharpe_30d:.2f}{sharpe_arrow}",
            "inline": True,
        },
        {
            "name": "Win Rate (recent)",
            "value": f"{snapshot.win_rate_recent:.0%}",
            "inline": True,
        },
        {
            "name": "Profit Factor",
            "value": "N/A"
            if snapshot.profit_factor == float("inf")
            else f"{snapshot.profit_factor:.2f}",
            "inline": True,
        },
        {
            "name": "Trades Total",
            "value": str(snapshot.total_closed_trades),
            "inline": True,
        },
    ]

    # Open positions
    if snapshot.open_positions:
        pos_lines: list[str] = []
        for pos in snapshot.open_positions:
            pnl_usd = pos.unrealized_pnl
            pos_lines.append(f"  {pos.direction}  {pos.symbol}  ${pnl_usd:+,.2f}")
        fields.append(
            {
                "name": f"Open Positions ({len(snapshot.open_positions)})",
                "value": "\n".join(pos_lines),
                "inline": False,
            }
        )
    else:
        fields.append({"name": "Open Positions", "value": "None", "inline": False})

    # Per-strategy breakdown
    if snapshot.strategy_breakdown:
        _status_icons = {"HEALTHY": "+", "WATCH": "~", "DEGRADING": "-"}
        breakdown_lines: list[str] = []
        for sp in snapshot.strategy_breakdown:
            icon = _status_icons.get(sp.status, "?")
            breakdown_lines.append(
                f"[{icon}] **{sp.strategy_name}**  "
                + f"Sharpe {sp.rolling_sharpe:.2f} | "
                + f"WR {sp.win_rate:.0%} | "
                + f"PnL ${sp.total_pnl:+,.0f} ({sp.trade_count})"
            )
        fields.append(
            {
                "name": "Strategy Breakdown (30d)",
                "value": "\n".join(breakdown_lines),
                "inline": False,
            }
        )

    cb_label = "ACTIVE" if snapshot.is_circuit_breaker_active else "OK"
    fields.append({"name": "CB Status", "value": cb_label, "inline": True})

    return {
        "title": "Strategy Health Report",
        "color": color,
        "fields": fields,
        "timestamp": snapshot.timestamp.isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_onchain_alert_embed(message: str, source: str) -> dict[str, Any]:
    """On-chain 데이터 수집 실패 알림 Embed.

    Args:
        message: 알림 메시지
        source: 소스 식별자

    Returns:
        Discord Embed dict
    """
    return {
        "title": f"On-chain Alert — {source}",
        "description": message,
        "color": _COLOR_YELLOW,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }
