"""DiscordBotService -- discord.py Bot + Slash Commands.

discord.py Client(WebSocket Gateway)를 기반으로:
1. 채널 라우팅: NotificationQueue에서 embed를 받아 지정 채널에 전송
2. Slash Commands: /status, /kill, /balance

Rules Applied:
    - #10 Python Standards: Async patterns, type hints
    - EDA 패턴: Graceful shutdown
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import discord
from discord import app_commands
from loguru import logger

from src.notification.models import ChannelRoute

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from prometheus_client import Gauge

    from src.eda.analytics import AnalyticsEngine
    from src.eda.portfolio_manager import EDAPortfolioManager
    from src.eda.risk_manager import EDARiskManager
    from src.notification.config import DiscordBotConfig
    from src.orchestrator.orchestrator import StrategyOrchestrator


def _read_gauge_value(gauge: Gauge) -> float:
    """Prometheus Gauge의 현재 값을 public API로 읽는다."""
    metrics = list(gauge.collect())
    if metrics and metrics[0].samples:
        return float(metrics[0].samples[0].value)
    return 0.0


@dataclass
class TradingContext:
    """Slash Commands에서 접근할 트레이딩 컨텍스트.

    Attributes:
        pm: 포지션, equity, cash 조회
        rm: drawdown, CB 상태 조회
        analytics: equity curve, trade records
        runner_shutdown: LiveRunner.request_shutdown() 콜백
        orchestrator: StrategyOrchestrator (Orchestrator 모드 시)
        report_trigger: ReportScheduler.trigger_daily_report() 콜백
        health_trigger: HealthCheckScheduler.trigger_health_check() 콜백
    """

    pm: EDAPortfolioManager
    rm: EDARiskManager
    analytics: AnalyticsEngine
    runner_shutdown: Callable[[], None]
    orchestrator: StrategyOrchestrator | None = None
    report_trigger: Callable[[], Coroutine[object, object, None]] | None = None
    health_trigger: Callable[[], Coroutine[object, object, None]] | None = None
    exchange_stop_mgr: object | None = None
    onchain_feed: object | None = None
    _active: bool = field(default=True, init=False)

    @property
    def is_active(self) -> bool:
        """시스템 활성 여부."""
        return self._active

    def deactivate(self) -> None:
        """시스템 비활성화 (kill 후)."""
        self._active = False


class _ConfirmView(discord.ui.View):
    """확인/취소 버튼 UI View (30초 timeout)."""

    confirmed: bool = False

    def __init__(self) -> None:
        super().__init__(timeout=30.0)

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.danger)
    async def confirm_btn(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[_ConfirmView],
    ) -> None:
        """확인 버튼 클릭."""
        self.confirmed = True
        self.stop()
        await interaction.response.edit_message(content="Confirmed.", view=None)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel_btn(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[_ConfirmView],
    ) -> None:
        """취소 버튼 클릭."""
        self.confirmed = False
        self.stop()
        await interaction.response.edit_message(content="Cancelled.", view=None)

    async def on_timeout(self) -> None:
        """30초 타임아웃."""
        self.confirmed = False
        self.stop()


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
        self._setup_read_commands()
        self._setup_action_commands()

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
                ChannelRoute.HEARTBEAT: self._config.heartbeat_channel_id,
                ChannelRoute.MARKET_REGIME: self._config.regime_channel_id,
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

    def _setup_read_commands(self) -> None:
        """조회형 Slash Commands 등록."""
        guild_id = self._config.guild_id
        guild_obj = discord.Object(id=guild_id) if guild_id else None

        async def status_cmd(interaction: discord.Interaction) -> None:
            await self._handle_status(interaction)

        async def balance_cmd(interaction: discord.Interaction) -> None:
            await self._handle_balance(interaction)

        async def strategies_cmd(interaction: discord.Interaction) -> None:
            await self._handle_strategies(interaction)

        async def report_cmd(interaction: discord.Interaction) -> None:
            await self._handle_report(interaction)

        async def health_cmd(interaction: discord.Interaction) -> None:
            await self._handle_health(interaction)

        async def metrics_cmd(interaction: discord.Interaction) -> None:
            await self._handle_metrics(interaction)

        async def onchain_cmd(interaction: discord.Interaction) -> None:
            await self._handle_onchain(interaction)

        self._tree.command(
            name="status", description="Open positions and equity status", guild=guild_obj
        )(status_cmd)
        self._tree.command(name="balance", description="Account balance overview", guild=guild_obj)(
            balance_cmd
        )
        self._tree.command(name="strategies", description="Strategy pod overview", guild=guild_obj)(
            strategies_cmd
        )
        self._tree.command(name="report", description="Trigger daily report now", guild=guild_obj)(
            report_cmd
        )
        self._tree.command(name="health", description="System health check", guild=guild_obj)(
            health_cmd
        )
        self._tree.command(name="metrics", description="Key metrics summary", guild=guild_obj)(
            metrics_cmd
        )
        self._tree.command(
            name="onchain", description="On-chain data source status", guild=guild_obj
        )(onchain_cmd)

    def _setup_action_commands(self) -> None:
        """액션형 Slash Commands 등록."""
        guild_id = self._config.guild_id
        guild_obj = discord.Object(id=guild_id) if guild_id else None

        async def kill_cmd(interaction: discord.Interaction) -> None:
            await self._handle_kill(interaction)

        @app_commands.describe(name="Strategy pod name")
        async def strategy_cmd(interaction: discord.Interaction, name: str) -> None:
            await self._handle_strategy_detail(interaction, name)

        @app_commands.describe(name="Strategy pod name to pause")
        async def pause_cmd(interaction: discord.Interaction, name: str) -> None:
            await self._handle_pause(interaction, name)

        @app_commands.describe(name="Strategy pod name to resume")
        async def resume_cmd(interaction: discord.Interaction, name: str) -> None:
            await self._handle_resume(interaction, name)

        self._tree.command(name="kill", description="Emergency shutdown", guild=guild_obj)(kill_cmd)
        self._tree.command(name="strategy", description="Strategy detail", guild=guild_obj)(
            strategy_cmd
        )
        self._tree.command(name="pause", description="Pause a strategy pod", guild=guild_obj)(
            pause_cmd
        )
        self._tree.command(name="resume", description="Resume a strategy pod", guild=guild_obj)(
            resume_cmd
        )

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

        # Safety stops 섹션
        if ctx.exchange_stop_mgr is not None:
            active_stops = ctx.exchange_stop_mgr.active_stops  # type: ignore[union-attr]
            if active_stops:
                stop_lines: list[str] = []
                for sym, state in active_stops.items():
                    fail_tag = (
                        f" ({state.placement_failures} fails)" if state.placement_failures else ""
                    )
                    stop_lines.append(
                        f"**{sym}** {state.position_side} @ ${state.stop_price:,.2f}{fail_tag}"
                    )
                embed.add_field(
                    name=f"Safety Stops ({len(active_stops)})",
                    value="\n".join(stop_lines),
                    inline=False,
                )
            else:
                embed.add_field(name="Safety Stops", value="None", inline=False)

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

    # ── L3-3 New Command Handlers ────────────────────────────────

    async def _handle_strategies(self, interaction: discord.Interaction) -> None:
        """``/strategies`` -- 전략 Pod 요약 목록."""
        ctx = self._trading_ctx
        if ctx is None or ctx.orchestrator is None:
            await interaction.response.send_message("Orchestrator not available.", ephemeral=True)
            return

        pods = ctx.orchestrator.pods
        embed = discord.Embed(title="Strategy Pods", color=0x3498DB)

        for pod in pods:
            perf = pod.performance
            status = "PAUSED" if pod.paused else pod.state.value
            value = (
                f"State: **{status}** | Sharpe: {perf.sharpe_ratio:.2f}\n"
                f"DD: {perf.max_drawdown:.1%} | WR: {perf.win_rate:.1%} | "
                f"Trades: {perf.trade_count}"
            )
            embed.add_field(name=pod.pod_id, value=value, inline=False)

        embed.set_footer(text=f"{len(pods)} pods total | MC-Coin-Bot")
        await interaction.response.send_message(embed=embed)

    async def _handle_strategy_detail(
        self,
        interaction: discord.Interaction,
        name: str,
    ) -> None:
        """``/strategy <name>`` -- 개별 전략 상세."""
        ctx = self._trading_ctx
        if ctx is None or ctx.orchestrator is None:
            await interaction.response.send_message("Orchestrator not available.", ephemeral=True)
            return

        pod = self._find_pod(name)
        if pod is None:
            await interaction.response.send_message(f"Pod `{name}` not found.", ephemeral=True)
            return

        perf = pod.performance
        status = "PAUSED" if pod.paused else pod.state.value
        embed = discord.Embed(title=f"Strategy: {pod.pod_id}", color=0x9B59B6)
        embed.add_field(name="State", value=status, inline=True)
        embed.add_field(name="Capital %", value=f"{pod.capital_fraction:.1%}", inline=True)
        embed.add_field(name="Sharpe", value=f"{perf.sharpe_ratio:.2f}", inline=True)
        embed.add_field(name="Max DD", value=f"{perf.max_drawdown:.1%}", inline=True)
        embed.add_field(name="Win Rate", value=f"{perf.win_rate:.1%}", inline=True)
        embed.add_field(name="Trades", value=str(perf.trade_count), inline=True)
        embed.add_field(name="Live Days", value=str(perf.live_days), inline=True)
        embed.add_field(name="Total Return", value=f"{perf.total_return:.2%}", inline=True)
        embed.add_field(name="Symbols", value=", ".join(pod.symbols), inline=False)

        # GBM result if available
        if ctx.orchestrator.lifecycle is not None:
            gbm = ctx.orchestrator.lifecycle.get_gbm_result(pod.pod_id)
            if gbm is not None:
                embed.add_field(
                    name="GBM DD",
                    value=f"{gbm.severity.value} | depth={gbm.current_depth:.1%} dur={gbm.current_duration_days}d",
                    inline=False,
                )

        embed.set_footer(text="MC-Coin-Bot")
        await interaction.response.send_message(embed=embed)

    async def _handle_pause(self, interaction: discord.Interaction, name: str) -> None:
        """``/pause <name>`` -- Pod 일시 중지 (확인 필요)."""
        ctx = self._trading_ctx
        if ctx is None or ctx.orchestrator is None:
            await interaction.response.send_message("Orchestrator not available.", ephemeral=True)
            return

        pod = self._find_pod(name)
        if pod is None:
            await interaction.response.send_message(f"Pod `{name}` not found.", ephemeral=True)
            return

        if pod.paused:
            await interaction.response.send_message(
                f"Pod `{name}` is already paused.", ephemeral=True
            )
            return

        view = _ConfirmView()
        await interaction.response.send_message(
            f"Pause strategy **{name}**? This will stop signal generation.",
            view=view,
        )
        await view.wait()
        if view.confirmed:
            pod.pause()
            await interaction.followup.send(f"Pod `{name}` paused.")
        elif not view.confirmed:
            await interaction.followup.send(f"Pause `{name}` cancelled/timed out.")

    async def _handle_resume(self, interaction: discord.Interaction, name: str) -> None:
        """``/resume <name>`` -- Pod 일시 중지 해제 (확인 필요)."""
        ctx = self._trading_ctx
        if ctx is None or ctx.orchestrator is None:
            await interaction.response.send_message("Orchestrator not available.", ephemeral=True)
            return

        pod = self._find_pod(name)
        if pod is None:
            await interaction.response.send_message(f"Pod `{name}` not found.", ephemeral=True)
            return

        if not pod.paused:
            await interaction.response.send_message(f"Pod `{name}` is not paused.", ephemeral=True)
            return

        view = _ConfirmView()
        await interaction.response.send_message(
            f"Resume strategy **{name}**?",
            view=view,
        )
        await view.wait()
        if view.confirmed:
            pod.resume()
            await interaction.followup.send(f"Pod `{name}` resumed.")
        elif not view.confirmed:
            await interaction.followup.send(f"Resume `{name}` cancelled/timed out.")

    async def _handle_report(self, interaction: discord.Interaction) -> None:
        """``/report`` -- 즉시 daily report 트리거."""
        ctx = self._trading_ctx
        if ctx is None or ctx.report_trigger is None:
            await interaction.response.send_message("Report trigger not available.", ephemeral=True)
            return

        await interaction.response.send_message("Triggering daily report...")
        await ctx.report_trigger()

    async def _handle_health(self, interaction: discord.Interaction) -> None:
        """``/health`` -- 시스템 건강 상태."""
        ctx = self._trading_ctx
        if ctx is None:
            await interaction.response.send_message(
                "Trading context not available.", ephemeral=True
            )
            return

        pm = ctx.pm
        rm = ctx.rm

        embed = discord.Embed(title="System Health", color=0x2ECC71)
        embed.add_field(name="Equity", value=f"${pm.total_equity:,.2f}", inline=True)
        embed.add_field(name="Cash", value=f"${pm.available_cash:,.2f}", inline=True)
        embed.add_field(name="Leverage", value=f"{pm.aggregate_leverage:.2f}x", inline=True)
        embed.add_field(name="Drawdown", value=f"{rm.current_drawdown:.2f}%", inline=True)
        cb_status = "ACTIVE" if rm.is_circuit_breaker_active else "OK"
        embed.add_field(name="Circuit Breaker", value=cb_status, inline=True)
        embed.add_field(name="Positions", value=str(pm.open_position_count), inline=True)

        # Trigger health check if available
        if ctx.health_trigger is not None:
            await ctx.health_trigger()
            embed.set_footer(text="Health check triggered | MC-Coin-Bot")
        else:
            embed.set_footer(text="MC-Coin-Bot")

        await interaction.response.send_message(embed=embed)

    async def _handle_metrics(self, interaction: discord.Interaction) -> None:
        """``/metrics`` -- 주요 Prometheus 메트릭 요약."""
        from src.monitoring.metrics import (
            equity_gauge,
        )

        embed = discord.Embed(title="Metrics Summary", color=0xF39C12)

        # Read gauge values via public collect() API
        equity_val = _read_gauge_value(equity_gauge)
        embed.add_field(name="Equity", value=f"${equity_val:,.2f}", inline=True)

        embed.set_footer(text="MC-Coin-Bot")
        await interaction.response.send_message(embed=embed)

    async def _handle_onchain(self, interaction: discord.Interaction) -> None:
        """``/onchain`` -- On-chain 데이터 소스 상태."""
        ctx = self._trading_ctx
        if ctx is None:
            await interaction.response.send_message(
                "Trading context not available.", ephemeral=True
            )
            return

        embed = discord.Embed(title="On-chain Data Status", color=0x9B59B6)

        # Cache 상태
        if ctx.onchain_feed is not None:
            health = ctx.onchain_feed.get_health_status()  # type: ignore[union-attr]
            embed.add_field(
                name="Cache",
                value=f"{health['symbols_cached']} symbols, {health['total_columns']} columns",
                inline=True,
            )
        else:
            embed.add_field(name="Cache", value="Not active", inline=True)

        # Source별 마지막 fetch 상태 (Prometheus gauge)
        try:
            import time as _time

            from src.monitoring.metrics import onchain_last_success_gauge

            metrics = list(onchain_last_success_gauge.collect())
            if metrics and metrics[0].samples:
                now = _time.time()
                lines: list[str] = []
                for sample in metrics[0].samples:
                    source = sample.labels.get("source", "?")
                    age_h = (now - sample.value) / 3600
                    icon = "[+]" if age_h < 48 else "[-]"  # noqa: PLR2004
                    lines.append(f"{icon} **{source}** — {age_h:.1f}h ago")
                if lines:
                    embed.add_field(
                        name="Sources",
                        value="\n".join(lines),
                        inline=False,
                    )
        except ImportError:
            pass

        embed.set_footer(text="MC-Coin-Bot")
        await interaction.response.send_message(embed=embed)

    def _find_pod(self, name: str) -> Any:
        """Orchestrator에서 Pod 조회."""
        ctx = self._trading_ctx
        if ctx is None or ctx.orchestrator is None:
            return None
        for pod in ctx.orchestrator.pods:
            if pod.pod_id == name:
                return pod
        return None

    async def send_embed(
        self,
        channel: ChannelRoute,
        embed_dict: dict[str, Any],
        files: Sequence[tuple[str, bytes]] = (),
    ) -> bool:
        """Embed를 지정 채널에 전송 (선택적 파일 첨부).

        Args:
            channel: 대상 채널 라우트
            embed_dict: Discord Embed dict
            files: 첨부 파일 목록 ((filename, data) 튜플)

        Returns:
            True이면 전송 성공
        """
        ch = self._channels.get(channel)
        if ch is None:
            logger.debug("Channel not available for route: {}", channel.value)
            return False

        try:
            embed = discord.Embed.from_dict(embed_dict)
            discord_files = [discord.File(io.BytesIO(data), filename=name) for name, data in files]
            await ch.send(
                embed=embed,
                files=discord_files if discord_files else discord.utils.MISSING,
            )
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
