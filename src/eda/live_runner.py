"""LiveRunner — 실시간 EDA 오케스트레이터.

LiveDataFeed(WebSocket) + EDA 컴포넌트를 조립하여 24/7 실행합니다.
Graceful shutdown (SIGTERM/SIGINT), signal handler, 무한 루프 지원.

Modes:
    - paper: LiveDataFeed + BacktestExecutor (시뮬레이션 체결)
    - shadow: LiveDataFeed + ShadowExecutor (로깅만, 체결 없음)

Rules Applied:
    - EDARunner와 동일한 컴포넌트 조립 패턴
    - DataFeed와 Executor만 교체 → 백테스트 → 라이브 전환
    - Graceful shutdown: feed.stop() → flush → bus.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from enum import StrEnum
from typing import TYPE_CHECKING

from loguru import logger

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType
from src.eda.analytics import AnalyticsEngine
from src.eda.executors import BacktestExecutor, ShadowExecutor
from src.eda.live_data_feed import LiveDataFeed
from src.eda.oms import OMS
from src.eda.portfolio_manager import EDAPortfolioManager
from src.eda.risk_manager import EDARiskManager
from src.eda.strategy_engine import StrategyEngine

if TYPE_CHECKING:
    from src.eda.persistence.database import Database
    from src.eda.persistence.state_manager import StateManager
    from src.eda.ports import ExecutorPort
    from src.exchange.binance_client import BinanceClient
    from src.monitoring.metrics import MetricsExporter
    from src.notification.bot import DiscordBotService
    from src.notification.config import DiscordBotConfig
    from src.notification.queue import NotificationQueue
    from src.notification.report_scheduler import ReportScheduler
    from src.portfolio.config import PortfolioManagerConfig
    from src.strategy.base import BaseStrategy

from dataclasses import dataclass

# 기본 상태 저장 주기 (초)
_DEFAULT_SAVE_INTERVAL = 300.0


@dataclass
class _DiscordTasks:
    """Discord 관련 task/service 번들."""

    bot_service: DiscordBotService
    notification_queue: NotificationQueue
    queue_task: asyncio.Task[None]
    bot_task: asyncio.Task[None]
    report_scheduler: ReportScheduler | None = None


class LiveMode(StrEnum):
    """LiveRunner 실행 모드."""

    PAPER = "paper"
    SHADOW = "shadow"


class LiveRunner:
    """Live EDA 오케스트레이터 — 24/7 실행.

    EDARunner와 동일한 컴포넌트 조립 패턴을 사용하되,
    LiveDataFeed(WebSocket)로 실시간 데이터를 수신합니다.

    Args:
        strategy: 전략 인스턴스
        feed: LiveDataFeed 인스턴스
        executor: ExecutorPort 구현체
        target_timeframe: 집계 목표 TF
        config: 포트폴리오 설정
        mode: 실행 모드 (paper/shadow)
        initial_capital: 초기 자본 (USD)
        asset_weights: 에셋별 가중치
        queue_size: EventBus 큐 크기
        db_path: SQLite 경로 (None이면 영속화 비활성)
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        feed: LiveDataFeed,
        executor: ExecutorPort,
        target_timeframe: str,
        config: PortfolioManagerConfig,
        mode: LiveMode,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        queue_size: int = 10000,
        db_path: str | None = None,
        discord_config: DiscordBotConfig | None = None,
        metrics_port: int = 0,
    ) -> None:
        self._strategy = strategy
        self._feed = feed
        self._executor = executor
        self._target_timeframe = target_timeframe
        self._config = config
        self._mode = mode
        self._initial_capital = initial_capital
        self._asset_weights = asset_weights
        self._queue_size = queue_size
        self._shutdown_event = asyncio.Event()
        self._db_path = db_path
        self._discord_config = discord_config
        self._metrics_port = metrics_port

    @classmethod
    def paper(
        cls,
        strategy: BaseStrategy,
        symbols: list[str],
        target_timeframe: str,
        config: PortfolioManagerConfig,
        client: BinanceClient,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        db_path: str | None = None,
        discord_config: DiscordBotConfig | None = None,
        metrics_port: int = 0,
    ) -> LiveRunner:
        """Paper 모드: LiveDataFeed + BacktestExecutor."""
        feed = LiveDataFeed(symbols, target_timeframe, client)
        executor = BacktestExecutor(cost_model=config.cost_model)
        return cls(
            strategy=strategy,
            feed=feed,
            executor=executor,
            target_timeframe=target_timeframe,
            config=config,
            mode=LiveMode.PAPER,
            initial_capital=initial_capital,
            asset_weights=asset_weights,
            db_path=db_path,
            discord_config=discord_config,
            metrics_port=metrics_port,
        )

    @classmethod
    def shadow(
        cls,
        strategy: BaseStrategy,
        symbols: list[str],
        target_timeframe: str,
        config: PortfolioManagerConfig,
        client: BinanceClient,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        db_path: str | None = None,
        discord_config: DiscordBotConfig | None = None,
        metrics_port: int = 0,
    ) -> LiveRunner:
        """Shadow 모드: LiveDataFeed + ShadowExecutor."""
        feed = LiveDataFeed(symbols, target_timeframe, client)
        executor = ShadowExecutor()
        return cls(
            strategy=strategy,
            feed=feed,
            executor=executor,
            target_timeframe=target_timeframe,
            config=config,
            mode=LiveMode.SHADOW,
            initial_capital=initial_capital,
            asset_weights=asset_weights,
            db_path=db_path,
            discord_config=discord_config,
            metrics_port=metrics_port,
        )

    async def run(self) -> None:
        """메인 루프. shutdown_event까지 실행."""
        from src.eda.persistence.database import Database
        from src.eda.persistence.trade_persistence import TradePersistence

        # 1. DB 초기화 (db_path가 있을 때만)
        db: Database | None = None
        if self._db_path:
            db = Database(self._db_path)
            await db.connect()

        try:
            # 2. 컴포넌트 생성
            bus = EventBus(queue_size=self._queue_size)

            strategy_engine = StrategyEngine(
                self._strategy, target_timeframe=self._target_timeframe
            )
            pm = EDAPortfolioManager(
                config=self._config,
                initial_capital=self._initial_capital,
                asset_weights=self._asset_weights,
                target_timeframe=self._target_timeframe,
            )
            rm = EDARiskManager(
                config=self._config,
                portfolio_manager=pm,
                max_order_size_usd=self._initial_capital * self._config.max_leverage_cap,
                enable_circuit_breaker=True,
            )
            oms = OMS(executor=self._executor, portfolio_manager=pm)
            analytics = AnalyticsEngine(initial_capital=self._initial_capital)

            # BacktestExecutor인 경우 bar 핸들러 등록 (Paper 모드)
            if isinstance(self._executor, BacktestExecutor):
                bt_executor = self._executor

                async def executor_bar_handler(event: AnyEvent) -> None:
                    assert isinstance(event, BarEvent)
                    bt_executor.on_bar(event)

                bus.subscribe(EventType.BAR, executor_bar_handler)

            # 3. 상태 복구 (DB 활성 시)
            state_mgr = await self._restore_state(db, pm, rm)

            # 4. 모든 컴포넌트 등록 (순서 중요)
            await strategy_engine.register(bus)
            await pm.register(bus)
            await rm.register(bus)
            await oms.register(bus)
            await analytics.register(bus)

            # 5. TradePersistence 등록 (analytics 이후)
            if db:
                persistence = TradePersistence(db, strategy_name=self._strategy.name)
                await persistence.register(bus)

            # 5.5. Prometheus MetricsExporter (선택적)
            metrics_exporter = await self._setup_metrics(bus)

            # 6. Discord Bot + NotificationEngine (선택적)
            discord_tasks = await self._setup_discord(bus, pm, rm, analytics)

            # 7. Signal handler
            self._setup_signal_handlers()

            # 8. 실행
            bus_task = asyncio.create_task(bus.start())
            feed_task = asyncio.create_task(self._feed.start(bus))

            # 주기적 상태 저장 task
            save_task: asyncio.Task[None] | None = None
            if state_mgr:
                save_task = asyncio.create_task(self._periodic_state_save(state_mgr, pm, rm))

            # 주기적 uptime 갱신 task
            uptime_task: asyncio.Task[None] | None = None
            if metrics_exporter is not None:
                uptime_task = asyncio.create_task(self._periodic_uptime_update(metrics_exporter))

            logger.info("LiveRunner started (mode={}, db={})", self._mode.value, self._db_path)

            # 8. Shutdown 대기
            await self._shutdown_event.wait()

            # 9. Graceful shutdown
            logger.warning("Initiating graceful shutdown...")

            # 주기적 task 취소
            for task in (save_task, uptime_task):
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

            await self._feed.stop()
            feed_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await feed_task

            await pm.flush_pending_signals()
            await bus.flush()
            await bus.stop()
            await bus_task

            # Discord Bot/Queue 종료
            await self._shutdown_discord(discord_tasks)

            # 최종 상태 저장
            if state_mgr:
                await state_mgr.save_all(pm, rm)
                logger.info("Final state saved")

            logger.info("LiveRunner stopped gracefully")
        finally:
            if db:
                await db.close()

    def request_shutdown(self) -> None:
        """외부에서 shutdown 요청 (테스트용)."""
        self._shutdown_event.set()

    @staticmethod
    async def _restore_state(
        db: Database | None,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
    ) -> StateManager | None:
        """DB에서 PM/RM 상태를 복구.

        Returns:
            StateManager 또는 None (DB 비활성 시)
        """
        if db is None:
            return None

        from src.eda.persistence.state_manager import StateManager

        state_mgr = StateManager(db)
        pm_state = await state_mgr.load_pm_state()
        if pm_state:
            pm.restore_state(pm_state)
            logger.info("PM state restored")
        rm_state = await state_mgr.load_rm_state()
        if rm_state:
            rm.restore_state(rm_state)
            logger.info("RM state restored")
        return state_mgr

    async def _setup_discord(
        self,
        bus: EventBus,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        analytics: AnalyticsEngine,
    ) -> _DiscordTasks | None:
        """Discord Bot + NotificationEngine 초기화 (선택적).

        Returns:
            _DiscordTasks 또는 None (비활성 시)
        """
        if not self._discord_config or not self._discord_config.is_bot_configured:
            return None

        from src.notification.bot import DiscordBotService, TradingContext
        from src.notification.engine import NotificationEngine
        from src.notification.queue import NotificationQueue

        bot_service = DiscordBotService(self._discord_config)
        notification_queue = NotificationQueue(bot_service)
        notification_engine = NotificationEngine(notification_queue)
        await notification_engine.register(bus)

        trading_ctx = TradingContext(
            pm=pm,
            rm=rm,
            analytics=analytics,
            runner_shutdown=self.request_shutdown,
        )
        bot_service.set_trading_context(trading_ctx)

        queue_task = asyncio.create_task(notification_queue.start())
        bot_task = asyncio.create_task(bot_service.start())

        # ReportScheduler 생성 + 시작
        from src.monitoring.chart_generator import ChartGenerator
        from src.notification.report_scheduler import ReportScheduler

        chart_gen = ChartGenerator()
        report_scheduler = ReportScheduler(
            queue=notification_queue,
            analytics=analytics,
            chart_gen=chart_gen,
            pm=pm,
        )
        await report_scheduler.start()

        logger.info("Discord Bot + NotificationEngine + ReportScheduler enabled")

        return _DiscordTasks(
            bot_service=bot_service,
            notification_queue=notification_queue,
            queue_task=queue_task,
            bot_task=bot_task,
            report_scheduler=report_scheduler,
        )

    @staticmethod
    async def _shutdown_discord(tasks: _DiscordTasks | None) -> None:
        """Discord Bot/Queue/ReportScheduler 정리."""
        if tasks is None:
            return
        if tasks.report_scheduler is not None:
            await tasks.report_scheduler.stop()
        await tasks.notification_queue.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await tasks.queue_task
        await tasks.bot_service.close()
        tasks.bot_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tasks.bot_task
        logger.info("Discord Bot stopped")

    def _setup_signal_handlers(self) -> None:
        """SIGTERM/SIGINT 핸들러 등록."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown, sig)

    def _handle_shutdown(self, sig: signal.Signals) -> None:
        """시그널 수신 시 shutdown 이벤트 설정."""
        logger.warning("Received {}, initiating graceful shutdown...", sig.name)
        self._shutdown_event.set()

    @staticmethod
    async def _periodic_state_save(
        state_mgr: StateManager,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        interval: float = _DEFAULT_SAVE_INTERVAL,
    ) -> None:
        """주기적으로 PM/RM 상태를 저장."""
        while True:
            await asyncio.sleep(interval)
            try:
                await state_mgr.save_all(pm, rm)
            except Exception:
                logger.exception("Periodic state save failed")

    async def _setup_metrics(self, bus: EventBus) -> MetricsExporter | None:
        """Prometheus MetricsExporter 초기화 (선택적).

        Returns:
            MetricsExporter 또는 None (비활성 시)
        """
        if self._metrics_port <= 0:
            return None

        from src.monitoring.metrics import MetricsExporter

        exporter = MetricsExporter(port=self._metrics_port)
        await exporter.register(bus)
        exporter.start_server()
        return exporter

    @staticmethod
    async def _periodic_uptime_update(
        exporter: MetricsExporter,
        interval: float = 30.0,
    ) -> None:
        """주기적으로 uptime gauge를 갱신."""
        while True:
            await asyncio.sleep(interval)
            exporter.update_uptime()

    @property
    def feed(self) -> LiveDataFeed:
        """LiveDataFeed 인스턴스."""
        return self._feed

    @property
    def mode(self) -> LiveMode:
        """실행 모드."""
        return self._mode
