"""DiscordBotService 테스트 (mock 기반)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.notification.bot import DiscordBotService, TradingContext
from src.notification.config import DiscordBotConfig
from src.notification.models import ChannelRoute


def _make_config(**overrides: object) -> DiscordBotConfig:
    defaults = {
        "bot_token": "test-token",
        "guild_id": 123456789,
        "trade_log_channel_id": 111,
        "alerts_channel_id": 222,
        "daily_report_channel_id": 333,
    }
    defaults.update(overrides)
    return DiscordBotConfig(**defaults)  # type: ignore[arg-type]


class TestDiscordBotConfig:
    def test_is_bot_configured_true(self) -> None:
        config = _make_config()
        assert config.is_bot_configured is True

    def test_is_bot_configured_no_token(self) -> None:
        config = _make_config(bot_token=None)
        assert config.is_bot_configured is False

    def test_is_bot_configured_no_guild(self) -> None:
        config = _make_config(guild_id=None)
        assert config.is_bot_configured is False


class TestDiscordBotService:
    def test_init(self) -> None:
        config = _make_config()
        bot = DiscordBotService(config)
        assert bot.is_ready is False

    def test_set_trading_context(self) -> None:
        config = _make_config()
        bot = DiscordBotService(config)

        pm = MagicMock()
        rm = MagicMock()
        analytics = MagicMock()
        shutdown = MagicMock()

        ctx = TradingContext(pm=pm, rm=rm, analytics=analytics, runner_shutdown=shutdown)
        bot.set_trading_context(ctx)
        assert bot._trading_ctx is ctx

    async def test_send_embed_no_channel(self) -> None:
        config = _make_config()
        bot = DiscordBotService(config)
        # 채널 캐싱 안 된 상태
        result = await bot.send_embed(ChannelRoute.TRADE_LOG, {"title": "Test"})
        assert result is False

    async def test_send_embed_with_channel(self) -> None:
        config = _make_config()
        bot = DiscordBotService(config)

        mock_channel = AsyncMock()
        bot._channels[ChannelRoute.TRADE_LOG] = mock_channel

        result = await bot.send_embed(ChannelRoute.TRADE_LOG, {"title": "Test", "type": "rich"})
        assert result is True
        mock_channel.send.assert_called_once()

    async def test_send_embed_with_files(self) -> None:
        config = _make_config()
        bot = DiscordBotService(config)

        mock_channel = AsyncMock()
        bot._channels[ChannelRoute.DAILY_REPORT] = mock_channel

        files = [("chart.png", b"\x89PNG_fake_data")]
        result = await bot.send_embed(
            ChannelRoute.DAILY_REPORT,
            {"title": "Report", "type": "rich"},
            files=files,
        )
        assert result is True
        call_kwargs = mock_channel.send.call_args.kwargs
        assert "files" in call_kwargs
        assert len(call_kwargs["files"]) == 1

    async def test_close_idempotent(self) -> None:
        config = _make_config()
        bot = DiscordBotService(config)

        # client.is_closed() = True이면 close 호출 안 함
        bot._client.is_closed = MagicMock(return_value=True)
        await bot.close()  # should not raise


class TestTradingContext:
    def test_active_by_default(self) -> None:
        ctx = TradingContext(
            pm=MagicMock(),
            rm=MagicMock(),
            analytics=MagicMock(),
            runner_shutdown=MagicMock(),
        )
        assert ctx.is_active is True

    def test_deactivate(self) -> None:
        ctx = TradingContext(
            pm=MagicMock(),
            rm=MagicMock(),
            analytics=MagicMock(),
            runner_shutdown=MagicMock(),
        )
        ctx.deactivate()
        assert ctx.is_active is False


class TestSlashCommandHandlers:
    @pytest.fixture
    def bot_with_context(self) -> DiscordBotService:
        config = _make_config()
        bot = DiscordBotService(config)
        pm = MagicMock()
        pm.total_equity = 10000.0
        pm.available_cash = 8000.0
        pm.aggregate_leverage = 1.5
        pm.open_position_count = 2
        pm.positions = {}

        rm = MagicMock()
        rm.current_drawdown = 3.5
        rm.is_circuit_breaker_active = False

        ctx = TradingContext(pm=pm, rm=rm, analytics=MagicMock(), runner_shutdown=MagicMock())
        bot.set_trading_context(ctx)
        return bot

    async def test_handle_status_no_context(self) -> None:
        config = _make_config()
        bot = DiscordBotService(config)

        interaction = AsyncMock()
        await bot._handle_status(interaction)
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert "not available" in str(call_kwargs)

    async def test_handle_status_with_context(self, bot_with_context: DiscordBotService) -> None:
        interaction = AsyncMock()
        await bot_with_context._handle_status(interaction)
        interaction.response.send_message.assert_called_once()
        # embed가 전달되었는지 확인
        call_kwargs = interaction.response.send_message.call_args
        assert "embed" in call_kwargs.kwargs

    async def test_handle_balance(self, bot_with_context: DiscordBotService) -> None:
        interaction = AsyncMock()
        await bot_with_context._handle_balance(interaction)
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert "embed" in call_kwargs.kwargs

    async def test_handle_kill(self, bot_with_context: DiscordBotService) -> None:
        interaction = AsyncMock()
        await bot_with_context._handle_kill(interaction)
        interaction.response.send_message.assert_called_once()
        # shutdown 콜백 호출 확인
        ctx = bot_with_context._trading_ctx
        assert ctx is not None
        ctx.runner_shutdown.assert_called_once()
        assert ctx.is_active is False
