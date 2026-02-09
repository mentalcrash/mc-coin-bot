"""DiscordBotConfig 테스트."""

from __future__ import annotations

from src.notification.config import DiscordBotConfig


class TestDiscordBotConfig:
    def test_default_values(self) -> None:
        """환경 변수 없이 생성 시 기본값 None."""
        config = DiscordBotConfig(
            _env_file=None,  # type: ignore[call-arg]
        )
        assert config.bot_token is None
        assert config.guild_id is None
        assert config.trade_log_channel_id is None
        assert config.alerts_channel_id is None
        assert config.daily_report_channel_id is None
        assert config.trade_webhook_url is None
        assert config.error_webhook_url is None
        assert config.report_webhook_url is None

    def test_is_bot_configured_both_set(self) -> None:
        config = DiscordBotConfig(
            bot_token="token123",
            guild_id=999,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert config.is_bot_configured is True

    def test_is_bot_configured_missing_token(self) -> None:
        config = DiscordBotConfig(
            guild_id=999,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert config.is_bot_configured is False

    def test_is_bot_configured_missing_guild(self) -> None:
        config = DiscordBotConfig(
            bot_token="token123",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert config.is_bot_configured is False

    def test_webhook_fields(self) -> None:
        config = DiscordBotConfig(
            trade_webhook_url="https://discord.com/api/webhooks/trade",
            error_webhook_url="https://discord.com/api/webhooks/error",
            report_webhook_url="https://discord.com/api/webhooks/report",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert config.trade_webhook_url is not None
        assert config.error_webhook_url is not None
        assert config.report_webhook_url is not None

    def test_channel_ids(self) -> None:
        config = DiscordBotConfig(
            trade_log_channel_id=111,
            alerts_channel_id=222,
            daily_report_channel_id=333,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert config.trade_log_channel_id == 111
        assert config.alerts_channel_id == 222
        assert config.daily_report_channel_id == 333
