"""Notification 데이터 모델.

NotificationQueue에서 사용하는 아이템 및 라우팅/심각도 열거형.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, ConfigDict
    - #10 Python Standards: StrEnum, modern typing
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class Severity(StrEnum):
    """알림 심각도."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ChannelRoute(StrEnum):
    """Discord 채널 라우팅 대상."""

    TRADE_LOG = "trade_log"
    ALERTS = "alerts"
    DAILY_REPORT = "daily_report"


class NotificationItem(BaseModel):
    """NotificationQueue에 넣을 알림 아이템.

    Attributes:
        severity: 알림 심각도
        channel: 전송 대상 채널
        embed: Discord Embed dict
        spam_key: SpamGuard 키 (None이면 항상 전송)
    """

    model_config = ConfigDict(frozen=True)

    severity: Severity
    channel: ChannelRoute
    embed: dict[str, Any]
    spam_key: str | None = None
