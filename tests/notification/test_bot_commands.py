"""Tests for L3-3 Discord Bot Commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.notification.bot import DiscordBotService, TradingContext, _ConfirmView
from src.notification.config import DiscordBotConfig
from src.orchestrator.models import LifecycleState


def _make_config() -> DiscordBotConfig:
    return DiscordBotConfig(
        bot_token="test-token",
        guild_id=123456789,
        trade_log_channel_id=111,
        alerts_channel_id=222,
        daily_report_channel_id=333,
    )


def _make_pod(pod_id: str = "ctrend", state: str = "production") -> MagicMock:
    """Mock StrategyPod."""
    pod = MagicMock()
    pod.pod_id = pod_id
    pod.paused = False
    pod.state = LifecycleState(state)
    pod.is_active = True
    pod.capital_fraction = 0.5
    pod.symbols = ("BTC/USDT", "ETH/USDT")

    perf = MagicMock()
    perf.sharpe_ratio = 1.5
    perf.max_drawdown = 0.15
    perf.win_rate = 0.55
    perf.trade_count = 42
    perf.live_days = 30
    perf.total_return = 0.12
    perf.calmar_ratio = 2.0
    pod.performance = perf

    return pod


def _make_orchestrator(pods: list[MagicMock] | None = None) -> MagicMock:
    """Mock StrategyOrchestrator."""
    orch = MagicMock()
    if pods is None:
        pods = [_make_pod("ctrend"), _make_pod("anchor-mom")]
    orch.pods = pods
    orch.lifecycle = MagicMock()
    orch.lifecycle.get_gbm_result.return_value = None
    return orch


def _make_bot_with_orchestrator(
    orchestrator: MagicMock | None = None,
) -> tuple[DiscordBotService, TradingContext]:
    """Orchestrator 포함 bot + context 생성."""
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

    orch = orchestrator or _make_orchestrator()
    report_trigger = AsyncMock()

    # Mock HealthDataCollector
    health_collector = MagicMock()
    system_health = MagicMock()
    system_health.total_equity = 10000.0
    system_health.available_cash = 8000.0
    system_health.aggregate_leverage = 1.5
    system_health.current_drawdown = 0.035
    system_health.is_circuit_breaker_active = False
    system_health.open_position_count = 2
    system_health.total_symbols = 8
    system_health.stale_symbol_count = 0
    system_health.uptime_seconds = 3600.0
    system_health.today_pnl = 50.0
    system_health.today_trades = 3
    system_health.max_queue_depth = 5
    system_health.is_notification_degraded = False
    system_health.safety_stop_count = 0
    system_health.safety_stop_failures = 0
    system_health.onchain_sources_ok = 0
    system_health.onchain_sources_total = 0
    system_health.onchain_cache_columns = 0
    system_health.peak_equity = 10500.0
    system_health.bars_emitted = 100
    system_health.events_dropped = 0
    system_health.timestamp = MagicMock()
    system_health.timestamp.isoformat.return_value = "2026-01-01T00:00:00+00:00"
    health_collector.collect_system_health.return_value = system_health

    ctx = TradingContext(
        pm=pm,
        rm=rm,
        analytics=MagicMock(),
        runner_shutdown=MagicMock(),
        orchestrator=orch,
        report_trigger=report_trigger,
        health_collector=health_collector,
    )
    bot.set_trading_context(ctx)
    return bot, ctx


class TestStrategiesCommand:
    """``/strategies`` 명령 테스트."""

    async def test_lists_pods(self) -> None:
        bot, _ctx = _make_bot_with_orchestrator()
        interaction = AsyncMock()
        await bot._handle_strategies(interaction)
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert "embed" in call_kwargs.kwargs

    async def test_no_orchestrator(self) -> None:
        config = _make_config()
        bot = DiscordBotService(config)
        ctx = TradingContext(
            pm=MagicMock(), rm=MagicMock(), analytics=MagicMock(), runner_shutdown=MagicMock()
        )
        bot.set_trading_context(ctx)

        interaction = AsyncMock()
        await bot._handle_strategies(interaction)
        call_args = interaction.response.send_message.call_args
        assert "not available" in str(call_args)


class TestStrategyCommand:
    """``/strategy <name>`` 명령 테스트."""

    async def test_found(self) -> None:
        bot, _ = _make_bot_with_orchestrator()
        interaction = AsyncMock()
        await bot._handle_strategy_detail(interaction, "ctrend")
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert "embed" in call_kwargs.kwargs

    async def test_not_found(self) -> None:
        bot, _ = _make_bot_with_orchestrator()
        interaction = AsyncMock()
        await bot._handle_strategy_detail(interaction, "nonexistent")
        call_args = interaction.response.send_message.call_args
        assert "not found" in str(call_args)


class TestPauseResumeCommand:
    """``/pause`` / ``/resume`` 명령 테스트."""

    async def test_pause_pod_not_found(self) -> None:
        bot, _ = _make_bot_with_orchestrator()
        interaction = AsyncMock()
        await bot._handle_pause(interaction, "nonexistent")
        assert "not found" in str(interaction.response.send_message.call_args)

    async def test_pause_already_paused(self) -> None:
        pod = _make_pod("ctrend")
        pod.paused = True
        orch = _make_orchestrator([pod])
        bot, _ = _make_bot_with_orchestrator(orch)

        interaction = AsyncMock()
        await bot._handle_pause(interaction, "ctrend")
        assert "already paused" in str(interaction.response.send_message.call_args)

    async def test_resume_not_paused(self) -> None:
        bot, _ = _make_bot_with_orchestrator()
        interaction = AsyncMock()
        await bot._handle_resume(interaction, "ctrend")
        assert "not paused" in str(interaction.response.send_message.call_args)


class TestReportCommand:
    """``/report`` 명령 테스트."""

    async def test_triggers_report(self) -> None:
        bot, ctx = _make_bot_with_orchestrator()
        interaction = AsyncMock()
        await bot._handle_report(interaction)
        interaction.response.send_message.assert_called_once()
        assert ctx.report_trigger is not None
        ctx.report_trigger.assert_called_once()

    async def test_no_trigger(self) -> None:
        config = _make_config()
        bot = DiscordBotService(config)
        ctx = TradingContext(
            pm=MagicMock(), rm=MagicMock(), analytics=MagicMock(), runner_shutdown=MagicMock()
        )
        bot.set_trading_context(ctx)

        interaction = AsyncMock()
        await bot._handle_report(interaction)
        assert "not available" in str(interaction.response.send_message.call_args)


class TestHealthMetrics:
    """``/health`` / ``/metrics`` 명령 테스트."""

    async def test_health_uses_collector(self) -> None:
        """health_collector가 있으면 collect_system_health → embed 반환."""
        bot, ctx = _make_bot_with_orchestrator()
        interaction = AsyncMock()
        await bot._handle_health(interaction)
        interaction.response.send_message.assert_called_once()
        assert ctx.health_collector is not None
        ctx.health_collector.collect_system_health.assert_called_once()

    async def test_health_fallback_without_collector(self) -> None:
        """health_collector=None → 간단 embed fallback."""
        config = _make_config()
        bot = DiscordBotService(config)
        ctx = TradingContext(
            pm=MagicMock(
                total_equity=10000.0,
                available_cash=8000.0,
                aggregate_leverage=1.5,
                open_position_count=2,
            ),
            rm=MagicMock(current_drawdown=3.5, is_circuit_breaker_active=False),
            analytics=MagicMock(),
            runner_shutdown=MagicMock(),
        )
        bot.set_trading_context(ctx)

        interaction = AsyncMock()
        await bot._handle_health(interaction)
        interaction.response.send_message.assert_called_once()
        call_kwargs = interaction.response.send_message.call_args
        assert "embed" in call_kwargs.kwargs

    async def test_metrics_responds(self) -> None:
        bot, _ = _make_bot_with_orchestrator()
        interaction = AsyncMock()
        await bot._handle_metrics(interaction)
        interaction.response.send_message.assert_called_once()


class TestConfirmationView:
    """_ConfirmView 테스트."""

    async def test_initial_state(self) -> None:
        view = _ConfirmView()
        assert view.confirmed is False
        assert view.timeout == 30.0

    async def test_on_timeout(self) -> None:
        view = _ConfirmView()
        await view.on_timeout()
        assert view.confirmed is False


class TestTradingContextExtended:
    """확장된 TradingContext 테스트."""

    def test_orchestrator_field(self) -> None:
        orch = MagicMock()
        ctx = TradingContext(
            pm=MagicMock(),
            rm=MagicMock(),
            analytics=MagicMock(),
            runner_shutdown=MagicMock(),
            orchestrator=orch,
        )
        assert ctx.orchestrator is orch

    def test_trigger_fields(self) -> None:
        report = AsyncMock()
        collector = MagicMock()
        ctx = TradingContext(
            pm=MagicMock(),
            rm=MagicMock(),
            analytics=MagicMock(),
            runner_shutdown=MagicMock(),
            report_trigger=report,
            health_collector=collector,
        )
        assert ctx.report_trigger is report
        assert ctx.health_collector is collector

    def test_default_none(self) -> None:
        ctx = TradingContext(
            pm=MagicMock(),
            rm=MagicMock(),
            analytics=MagicMock(),
            runner_shutdown=MagicMock(),
        )
        assert ctx.orchestrator is None
        assert ctx.report_trigger is None
        assert ctx.health_collector is None
