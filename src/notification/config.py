"""Discord Bot 통합 설정.

Bot (WebSocket Gateway + Slash Commands) 및 Webhook (fallback) 설정을 관리합니다.

Rules Applied:
    - #11 Pydantic Modeling: Settings management, env_prefix
    - #10 Python Standards: Modern typing, property
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DiscordBotConfig(BaseSettings):
    """Discord 통합 설정 -- Bot + Webhook 모두 관리.

    환경 변수 DISCORD_* 로 설정합니다.

    Attributes:
        bot_token: Discord Bot 토큰 (Bot 모드 필수)
        guild_id: Discord 서버 ID
        trade_log_channel_id: 거래 알림 채널 ID
        alerts_channel_id: 리스크/CB 알림 채널 ID
        daily_report_channel_id: 일일 보고서 채널 ID
        trade_webhook_url: Webhook URL (fallback)
        error_webhook_url: 에러 Webhook URL (fallback)
        report_webhook_url: 보고서 Webhook URL (fallback)
    """

    model_config = SettingsConfigDict(
        env_prefix="DISCORD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Bot 설정 (primary)
    bot_token: str | None = Field(default=None, description="Discord Bot 토큰")
    guild_id: int | None = Field(default=None, description="Discord 서버 ID")
    trade_log_channel_id: int | None = Field(default=None, description="거래 알림 채널 ID")
    alerts_channel_id: int | None = Field(default=None, description="리스크/CB 알림 채널 ID")
    daily_report_channel_id: int | None = Field(default=None, description="일일 보고서 채널 ID")

    # Webhook 설정 (fallback -- 기존 DiscordChannelConfig 호환)
    trade_webhook_url: str | None = Field(default=None, description="거래 Webhook URL")
    error_webhook_url: str | None = Field(default=None, description="에러 Webhook URL")
    report_webhook_url: str | None = Field(default=None, description="보고서 Webhook URL")

    @property
    def is_bot_configured(self) -> bool:
        """Bot 모드 사용 가능 여부."""
        return self.bot_token is not None and self.guild_id is not None
