"""DiscordBotService -- discord.py Bot + Slash Commands.

discord.py Client(WebSocket Gateway)를 기반으로:
1. 채널 라우팅: NotificationQueue에서 embed를 받아 지정 채널에 전송
2. Slash Commands: /status, /kill, /balance

Rules Applied:
    - #10 Python Standards: Async patterns, type hints
    - EDA 패턴: Graceful shutdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import discord
from discord import app_commands
from loguru import logger

from src.notification.models import ChannelRoute

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.eda.analytics import AnalyticsEngine
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.eda.risk_manager import EDARiskManager
    from src.notification.config import DiscordBotConfig


@dataclass
class TradingContext:
    """Slash Commands에서 접근할 트레이딩 컨텍스트.

    Attributes:
        pm: 포지션, equity, cash 조회
        rm: drawdown, CB 상태 조회
        analytics: equity curve, trade records
        runner_shutdown: LiveRunner.request_shutdown() 콜백
    """

    pm: EDAPortfolioManager
    rm: EDARiskManager
    analytics: AnalyticsEngine
    runner_shutdown: Callable[[], None]
    _active: bool = field(default=True, init=False)

    @property
    def is_active(self) -> bool:
        """시스템 활성 여부."""
        return self._active

    def deactivate(self) -> None:
        """시스템 비활성화 (kill 후)."""
        self._active = False


class DiscordBotService:
    """discord.py Bot -- 채널 알림 + Slash Commands.

    Args:
        config: DiscordBotConfig (bot_token, guild_id, channel IDs)
    """

    def __init__(self, config: DiscordBotConfig) -> None:
        self._config = config
        intents = discord.Intents.default()
        self._client = discord.Client(intents=intents)
        self._tree = app_commands.CommandTree(self._client)
        self._channels: dict[ChannelRoute, discord.TextChannel] = {}
        self._trading_ctx: TradingContext | None = None
        self._ready = False
        self._setup_events()
        self._setup_commands()

    def set_trading_context(self, ctx: TradingContext) -> None:
        """트레이딩 컨텍스트 설정 (Slash Commands에서 사용).

        Args:
            ctx: TradingContext 인스턴스
        """
        self._trading_ctx = ctx

    @property
    def is_ready(self) -> bool:
        """Bot이 Discord에 연결되어 ready 상태인지."""
        return self._ready

    def _setup_events(self) -> None:
        """discord.py 이벤트 핸들러 등록."""

        async def on_ready() -> None:
            guild_id = self._config.guild_id
            if guild_id is None:
                logger.error("Discord guild_id not configured")
                return

            guild = self._client.get_guild(guild_id)
            if guild is None:
                logger.error("Discord guild not found: {}", guild_id)
                return

            # 채널 캐싱
            channel_map: dict[ChannelRoute, int | None] = {
                ChannelRoute.TRADE_LOG: self._config.trade_log_channel_id,
                ChannelRoute.ALERTS: self._config.alerts_channel_id,
                ChannelRoute.DAILY_REPORT: self._config.daily_report_channel_id,
            }
            for route, ch_id in channel_map.items():
                if ch_id is not None:
                    ch = guild.get_channel(ch_id)
                    if isinstance(ch, discord.TextChannel):
                        self._channels[route] = ch
                        logger.debug("Discord channel cached: {} -> #{}", route.value, ch.name)

            # Slash commands sync
            try:
                await self._tree.sync(guild=discord.Object(id=guild_id))
                logger.info(
                    "Discord Bot ready (guild={}, channels={})",
                    guild.name,
                    len(self._channels),
                )
            except discord.HTTPException:
                logger.exception("Failed to sync slash commands")

            self._ready = True

        self._client.event(on_ready)

    def _setup_commands(self) -> None:
        """Slash Commands 등록 (/status, /kill, /balance)."""
        guild_id = self._config.guild_id
        guild_obj = discord.Object(id=guild_id) if guild_id else None

        async def status_cmd(interaction: discord.Interaction) -> None:
            await self._handle_status(interaction)

        async def kill_cmd(interaction: discord.Interaction) -> None:
            await self._handle_kill(interaction)

        async def balance_cmd(interaction: discord.Interaction) -> None:
            await self._handle_balance(interaction)

        self._tree.command(
            name="status",
            description="Open positions and equity status",
            guild=guild_obj,
        )(status_cmd)
        self._tree.command(
            name="kill",
            description="Emergency: trigger circuit breaker and shutdown",
            guild=guild_obj,
        )(kill_cmd)
        self._tree.command(
            name="balance",
            description="Account balance overview",
            guild=guild_obj,
        )(balance_cmd)

    async def _handle_status(self, interaction: discord.Interaction) -> None:
        """``/status`` -- 오픈 포지션 + equity."""
        if self._trading_ctx is None:
            await interaction.response.send_message(
                "Trading context not available.", ephemeral=True
            )
            return

        ctx = self._trading_ctx
        pm = ctx.pm
        positions = pm.positions

        embed = discord.Embed(
            title="Trading Status",
            color=0x3498DB,
        )
        embed.add_field(name="Total Equity", value=f"${pm.total_equity:,.2f}", inline=True)
        embed.add_field(name="Available Cash", value=f"${pm.available_cash:,.2f}", inline=True)
        embed.add_field(name="Leverage", value=f"{pm.aggregate_leverage:.2f}x", inline=True)

        rm = ctx.rm
        embed.add_field(name="Drawdown", value=f"{rm.current_drawdown:.2f}%", inline=True)
        cb_status = "ACTIVE" if rm.is_circuit_breaker_active else "OK"
        embed.add_field(name="Circuit Breaker", value=cb_status, inline=True)

        if positions:
            pos_lines: list[str] = []
            for sym, pos in positions.items():
                direction = pos.direction.name
                pnl = pos.unrealized_pnl
                pos_lines.append(
                    f"**{sym}** {direction} | Size: {pos.size:.6f} | PnL: ${pnl:+,.2f}"
                )
            embed.add_field(
                name=f"Open Positions ({len(positions)})",
                value="\n".join(pos_lines),
                inline=False,
            )
        else:
            embed.add_field(name="Open Positions", value="None", inline=False)

        embed.set_footer(text="MC-Coin-Bot")
        await interaction.response.send_message(embed=embed)

    async def _handle_kill(self, interaction: discord.Interaction) -> None:
        """``/kill`` -- 긴급 전체 청산 + shutdown."""
        if self._trading_ctx is None:
            await interaction.response.send_message(
                "Trading context not available.", ephemeral=True
            )
            return

        ctx = self._trading_ctx

        embed = discord.Embed(
            title="KILL SWITCH ACTIVATED",
            description="Initiating emergency shutdown...",
            color=0xED4245,
        )
        embed.set_footer(text="MC-Coin-Bot")
        await interaction.response.send_message(embed=embed)

        ctx.deactivate()
        ctx.runner_shutdown()

    async def _handle_balance(self, interaction: discord.Interaction) -> None:
        """``/balance`` -- 잔고 조회."""
        if self._trading_ctx is None:
            await interaction.response.send_message(
                "Trading context not available.", ephemeral=True
            )
            return

        ctx = self._trading_ctx
        pm = ctx.pm

        embed = discord.Embed(
            title="Account Balance",
            color=0x57F287,
        )
        embed.add_field(name="Total Equity", value=f"${pm.total_equity:,.2f}", inline=True)
        embed.add_field(name="Available Cash", value=f"${pm.available_cash:,.2f}", inline=True)
        embed.add_field(
            name="Open Positions",
            value=str(pm.open_position_count),
            inline=True,
        )
        embed.add_field(
            name="Aggregate Leverage",
            value=f"{pm.aggregate_leverage:.2f}x",
            inline=True,
        )
        embed.set_footer(text="MC-Coin-Bot")
        await interaction.response.send_message(embed=embed)

    async def send_embed(self, channel: ChannelRoute, embed_dict: dict[str, Any]) -> bool:
        """Embed를 지정 채널에 전송.

        Args:
            channel: 대상 채널 라우트
            embed_dict: Discord Embed dict

        Returns:
            True이면 전송 성공
        """
        ch = self._channels.get(channel)
        if ch is None:
            logger.debug("Channel not available for route: {}", channel.value)
            return False

        try:
            embed = discord.Embed.from_dict(embed_dict)
            await ch.send(embed=embed)
        except discord.HTTPException:
            logger.exception("Failed to send embed to {}", channel.value)
            return False
        else:
            return True

    async def start(self) -> None:
        """Bot 시작 (asyncio task로 실행).

        Raises:
            discord.LoginFailure: 잘못된 토큰
        """
        token = self._config.bot_token
        if token is None:
            logger.error("Discord bot_token not configured, skipping bot start")
            return
        await self._client.start(token)

    async def close(self) -> None:
        """Bot 종료."""
        if not self._client.is_closed():
            await self._client.close()
