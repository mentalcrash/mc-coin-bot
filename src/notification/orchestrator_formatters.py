"""Orchestrator → Discord Embed 변환 함수.

Orchestrator 이벤트(생애주기 전이, 리밸런스, 리스크 경고, 일일 리포트)를
Discord Embed dict로 변환하는 순수 함수 모듈입니다.

health_formatters.py 패턴을 따릅니다.

Rules Applied:
    - #10 Python Standards: Pure functions, type hints
    - #22 Notification Standards: Rich Embeds, color segmentation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

# ── Color Constants ──────────────────────────────────────────────

_COLOR_GREEN = 0x57F287  # graduation, recovery
_COLOR_YELLOW = 0xFFFF00  # WARNING 전이
_COLOR_ORANGE = 0xE67E22  # PROBATION 전이
_COLOR_RED = 0xED4245  # RETIRED 전이, critical risk
_COLOR_BLUE = 0x3498DB  # info, rebalance

_FOOTER_TEXT = "MC-Coin-Bot Orchestrator"

# Daily report drawdown 임계값
_DD_GREEN_THRESHOLD = 0.05
_DD_YELLOW_THRESHOLD = 0.10

# 생애주기 상태별 색상 매핑
_LIFECYCLE_COLORS: dict[str, int] = {
    "incubation": _COLOR_BLUE,
    "production": _COLOR_GREEN,
    "warning": _COLOR_YELLOW,
    "probation": _COLOR_ORANGE,
    "retired": _COLOR_RED,
}

# 생애주기 상태별 이모지 매핑
_LIFECYCLE_EMOJI: dict[str, str] = {
    "incubation": "🥚",
    "production": "🚀",
    "warning": "⚠️",
    "probation": "🔶",
    "retired": "💀",
}


def format_lifecycle_transition_embed(
    pod_id: str,
    from_state: str,
    to_state: str,
    timestamp: str,
    performance_summary: dict[str, object] | None = None,
) -> dict[str, Any]:
    """생애주기 상태 전이 → Discord Embed dict.

    Args:
        pod_id: Pod 식별자
        from_state: 이전 상태
        to_state: 새 상태
        timestamp: 전이 시각 (ISO 문자열)
        performance_summary: 성과 요약 (선택)

    Returns:
        Discord Embed dict
    """
    color = _LIFECYCLE_COLORS.get(to_state, _COLOR_BLUE)
    emoji = _LIFECYCLE_EMOJI.get(to_state, "📋")

    fields: list[dict[str, Any]] = [
        {"name": "Pod", "value": pod_id, "inline": True},
        {"name": "Transition", "value": f"{from_state} → {to_state}", "inline": True},
    ]

    if performance_summary:
        perf_lines: list[str] = []
        for key, value in performance_summary.items():
            if isinstance(value, float):
                perf_lines.append(f"**{key}**: {value:.4f}")
            else:
                perf_lines.append(f"**{key}**: {value}")
        if perf_lines:
            fields.append({"name": "Performance", "value": "\n".join(perf_lines), "inline": False})

    return {
        "title": f"{emoji} Pod Lifecycle — {to_state.upper()}",
        "color": color,
        "fields": fields,
        "timestamp": timestamp,
        "footer": {"text": _FOOTER_TEXT},
    }


def format_capital_rebalance_embed(
    timestamp: str,
    allocations: dict[str, float],
    trigger_reason: str,
) -> dict[str, Any]:
    """자본 리밸런스 → Discord Embed dict.

    Args:
        timestamp: 리밸런스 시각 (ISO 문자열)
        allocations: {pod_id: fraction} 배분 결과
        trigger_reason: 트리거 사유 (calendar/threshold/hybrid)

    Returns:
        Discord Embed dict
    """
    alloc_lines: list[str] = []
    for pod_id, fraction in sorted(allocations.items()):
        bar_len = int(fraction * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        alloc_lines.append(f"`{pod_id:<16}` {bar} {fraction:.1%}")

    description = "\n".join(alloc_lines) if alloc_lines else "No active pods"

    return {
        "title": "⚖️ Capital Rebalance",
        "color": _COLOR_BLUE,
        "description": description,
        "fields": [
            {"name": "Trigger", "value": trigger_reason, "inline": True},
            {"name": "Pods", "value": str(len(allocations)), "inline": True},
        ],
        "timestamp": timestamp,
        "footer": {"text": _FOOTER_TEXT},
    }


def format_portfolio_risk_alert_embed(
    alert_type: str,
    severity: str,
    message: str,
    current_value: float,
    threshold: float,
    pod_id: str | None = None,
) -> dict[str, Any]:
    """포트폴리오 리스크 경고 → Discord Embed dict.

    Args:
        alert_type: 경고 유형
        severity: 심각도 (warning/critical)
        message: 상세 메시지
        current_value: 현재 측정값
        threshold: 설정 임계값
        pod_id: 관련 Pod ID (None = 포트폴리오 전체)

    Returns:
        Discord Embed dict
    """
    color = _COLOR_RED if severity == "critical" else _COLOR_YELLOW

    fields: list[dict[str, Any]] = [
        {"name": "Type", "value": alert_type, "inline": True},
        {"name": "Severity", "value": severity.upper(), "inline": True},
        {"name": "Current", "value": f"{current_value:.4f}", "inline": True},
        {"name": "Threshold", "value": f"{threshold:.4f}", "inline": True},
    ]

    if pod_id is not None:
        fields.append({"name": "Pod", "value": pod_id, "inline": True})

    return {
        "title": f"🚨 Risk Alert — {alert_type}",
        "color": color,
        "description": message,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_daily_orchestrator_report_embed(
    pod_summaries: list[dict[str, object]],
    total_equity: float,
    effective_n: float,
    avg_correlation: float,
    portfolio_dd: float,
    gross_leverage: float,
) -> dict[str, Any]:
    """Orchestrator 일일 리포트 → Discord Embed dict.

    Args:
        pod_summaries: Pod 요약 리스트
        total_equity: 총 자본
        effective_n: 유효 분산 수
        avg_correlation: 평균 상관계수
        portfolio_dd: 포트폴리오 현재 낙폭
        gross_leverage: 총 레버리지

    Returns:
        Discord Embed dict
    """
    # Pod 테이블
    table_lines: list[str] = []
    table_lines.append("`Pod              State       Alloc   Days`")
    table_lines.append("`─────────────────────────────────────────`")
    for summary in pod_summaries:
        pid = str(summary.get("pod_id", ""))
        state = str(summary.get("state", ""))
        raw_frac = summary.get("capital_fraction", 0.0)
        frac = float(raw_frac) if isinstance(raw_frac, (int, float)) else 0.0
        raw_days = summary.get("live_days", 0)
        days = int(raw_days) if isinstance(raw_days, (int, float)) else 0
        table_lines.append(f"`{pid:<16} {state:<11} {frac:>5.1%}  {days:>4}d`")

    description = "\n".join(table_lines) if table_lines else "No pods"

    color = (
        _COLOR_GREEN
        if portfolio_dd < _DD_GREEN_THRESHOLD
        else (_COLOR_YELLOW if portfolio_dd < _DD_YELLOW_THRESHOLD else _COLOR_RED)
    )

    return {
        "title": "📊 Daily Orchestrator Report",
        "color": color,
        "description": description,
        "fields": [
            {"name": "Total Equity", "value": f"${total_equity:,.0f}", "inline": True},
            {"name": "Effective N", "value": f"{effective_n:.2f}", "inline": True},
            {"name": "Avg Correlation", "value": f"{avg_correlation:.2%}", "inline": True},
            {"name": "Drawdown", "value": f"-{portfolio_dd:.1%}", "inline": True},
            {"name": "Gross Leverage", "value": f"{gross_leverage:.2f}x", "inline": True},
            {"name": "Active Pods", "value": str(len(pod_summaries)), "inline": True},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }
