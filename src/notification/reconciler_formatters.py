"""Position Reconciliation Discord Embed 포매터.

Drift/Balance 불일치 감지 시 Discord ALERTS 채널에 전송할 embed를 생성합니다.

Rules Applied:
    - #10 Python Standards: Pure functions, type hints
    - #22 Notification Standards: Rich Embeds, color segmentation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# Discord Embed 색상 코드 (decimal)
_COLOR_ORANGE = 0xE67E22
_COLOR_RED = 0xED4245
_COLOR_YELLOW = 0xFFFF00

_FOOTER_TEXT = "MC-Coin-Bot"

# Drift severity 임계값
# Position drift 10%: mark-to-market 변동으로 노이즈가 높아 넓은 임계값 사용
_CRITICAL_DRIFT_PCT = 10.0
# Balance drift 5%: 잔고는 안정적이므로 타이트한 임계값
_BALANCE_CRITICAL_PCT = 5.0


@dataclass(frozen=True)
class DriftDetail:
    """심볼별 포지션 drift 상세 정보.

    Attributes:
        symbol: 거래 심볼
        pm_size: PM 포지션 크기
        pm_side: PM 방향 ("LONG" | "SHORT" | "FLAT")
        exchange_size: 거래소 포지션 크기
        exchange_side: 거래소 방향
        drift_pct: Drift 비율 (%)
        is_orphan: 한쪽만 포지션 보유 여부
        auto_corrected: Auto-correction 적용 여부
    """

    symbol: str
    pm_size: float
    pm_side: str
    exchange_size: float
    exchange_side: str
    drift_pct: float
    is_orphan: bool = False
    auto_corrected: bool = False


def format_position_drift_embed(drifts: list[DriftDetail]) -> dict[str, Any]:
    """Position drift 목록 → Discord embed dict.

    Args:
        drifts: DriftDetail 리스트

    Returns:
        Discord Embed dict (ORANGE 일반 / RED orphan 또는 10%+ drift)
    """
    has_critical = any(d.is_orphan or d.drift_pct >= _CRITICAL_DRIFT_PCT for d in drifts)
    color = _COLOR_RED if has_critical else _COLOR_ORANGE

    fields: list[dict[str, Any]] = []
    for d in drifts:
        action = "Auto-corrected" if d.auto_corrected else "Manual review needed"
        label = f"{'ORPHAN ' if d.is_orphan else ''}{d.symbol}"
        value = (
            f"PM: {d.pm_size:.6f} ({d.pm_side})\n"
            f"Exchange: {d.exchange_size:.6f} ({d.exchange_side})\n"
            f"Drift: {d.drift_pct:.1f}% | {action}"
        )
        fields.append({"name": label, "value": value, "inline": False})

    return {
        "title": f"Position Drift Detected ({len(drifts)} symbol{'s' if len(drifts) != 1 else ''})",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }


def format_balance_drift_embed(
    *,
    pm_equity: float,
    exchange_equity: float,
    drift_pct: float,
) -> dict[str, Any]:
    """Balance drift → Discord embed dict.

    Args:
        pm_equity: PM equity
        exchange_equity: 거래소 equity
        drift_pct: Drift 비율 (%)

    Returns:
        Discord Embed dict (YELLOW 2~5% / RED 5%+)
    """
    color = _COLOR_RED if drift_pct >= _BALANCE_CRITICAL_PCT else _COLOR_YELLOW

    level = "CRITICAL" if drift_pct >= _BALANCE_CRITICAL_PCT else "WARNING"

    return {
        "title": f"Balance Drift {level}",
        "color": color,
        "fields": [
            {"name": "PM Equity", "value": f"${pm_equity:,.0f}", "inline": True},
            {"name": "Exchange Equity", "value": f"${exchange_equity:,.0f}", "inline": True},
            {"name": "Drift", "value": f"{drift_pct:.1f}%", "inline": True},
        ],
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {"text": _FOOTER_TEXT},
    }
