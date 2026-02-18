"""LiveRunner — 실시간 EDA 오케스트레이터.

LiveDataFeed(WebSocket) + EDA 컴포넌트를 조립하여 24/7 실행합니다.
Graceful shutdown (SIGTERM/SIGINT), signal handler, 무한 루프 지원.

Modes:
    - paper: LiveDataFeed + BacktestExecutor (시뮬레이션 체결)
    - shadow: LiveDataFeed + ShadowExecutor (로깅만, 체결 없음)
    - live: LiveDataFeed + LiveExecutor (Binance Futures 실주문)

Rules Applied:
    - EDARunner와 동일한 컴포넌트 조립 패턴
    - DataFeed와 Executor만 교체 → 백테스트 → 라이브 전환
    - Graceful shutdown: feed.stop() → flush → bus.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType, HeartbeatEvent
from src.eda.analytics import AnalyticsEngine
from src.eda.executors import BacktestExecutor, LiveExecutor, ShadowExecutor
from src.eda.live_data_feed import LiveDataFeed
from src.eda.oms import OMS
from src.eda.portfolio_manager import EDAPortfolioManager
from src.eda.risk_manager import EDARiskManager
from src.eda.strategy_engine import StrategyEngine
from src.logging.tracing import setup_tracing, shutdown_tracing
from src.notification.models import ChannelRoute, NotificationItem, Severity

if TYPE_CHECKING:
    from src.eda.persistence.database import Database
    from src.eda.persistence.state_manager import StateManager
    from src.eda.ports import ExecutorPort
    from src.eda.reconciler import PositionReconciler
    from src.exchange.binance_client import BinanceClient
    from src.exchange.binance_futures_client import BinanceFuturesClient
    from src.market.feature_store import FeatureStore, FeatureStoreConfig
    from src.monitoring.metrics import MetricsExporter
    from src.notification.bot import DiscordBotService
    from src.notification.config import DiscordBotConfig
    from src.notification.health_scheduler import HealthCheckScheduler
    from src.notification.queue import NotificationQueue
    from src.notification.report_scheduler import ReportScheduler
    from src.orchestrator.config import OrchestratorConfig
    from src.orchestrator.metrics import OrchestratorMetrics
    from src.orchestrator.orchestrator import StrategyOrchestrator
    from src.portfolio.config import PortfolioManagerConfig
    from src.regime.service import RegimeService, RegimeServiceConfig
    from src.strategy.base import BaseStrategy

from dataclasses import dataclass

# 기본 상태 저장 주기 (초)
_DEFAULT_SAVE_INTERVAL = 300.0

# warmup 시 최소 필요 캔들 수 (마지막 1개 제거 후 1개 이상)
_MIN_WARMUP_CANDLES = 2


@dataclass
class _DiscordTasks:
    """Discord 관련 task/service 번들."""

    bot_service: DiscordBotService
    notification_queue: NotificationQueue
    queue_task: asyncio.Task[None]
    bot_task: asyncio.Task[None]
    report_scheduler: ReportScheduler | None = None
    health_scheduler: HealthCheckScheduler | None = None


class LiveMode(StrEnum):
    """LiveRunner 실행 모드."""

    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"


# Reconciler 검증 주기 (초)
_RECONCILER_INTERVAL = 60.0


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
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
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
        self._regime_config = regime_config
        self._feature_store_config = feature_store_config
        self._client: BinanceClient | None = None
        self._futures_client: BinanceFuturesClient | None = None
        self._symbols: list[str] = []
        self._derivatives_feed: Any = None  # LiveDerivativesFeed | None
        self._onchain_feed: Any = None  # LiveOnchainFeed | None
        self._macro_feed: Any = None  # LiveMacroFeed | None
        self._options_feed: Any = None  # LiveOptionsFeed | None
        self._deriv_ext_feed: Any = None  # LiveDerivExtFeed | None
        self._orchestrator: StrategyOrchestrator | None = None
        self._start_time = time.monotonic()

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
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
    ) -> LiveRunner:
        """Paper 모드: LiveDataFeed + BacktestExecutor."""
        feed = LiveDataFeed(symbols, target_timeframe, client)
        executor = BacktestExecutor(cost_model=config.cost_model)
        runner = cls(
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
            regime_config=regime_config,
            feature_store_config=feature_store_config,
        )
        runner._client = client
        runner._symbols = symbols
        runner._init_onchain_feed(symbols)
        runner._init_macro_feed()
        runner._init_options_feed()
        runner._init_deriv_ext_feed(symbols)
        return runner

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
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
    ) -> LiveRunner:
        """Shadow 모드: LiveDataFeed + ShadowExecutor."""
        feed = LiveDataFeed(symbols, target_timeframe, client)
        executor = ShadowExecutor()
        runner = cls(
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
            regime_config=regime_config,
            feature_store_config=feature_store_config,
        )
        runner._client = client
        runner._symbols = symbols
        runner._init_onchain_feed(symbols)
        runner._init_macro_feed()
        runner._init_options_feed()
        runner._init_deriv_ext_feed(symbols)
        return runner

    @classmethod
    def live(
        cls,
        strategy: BaseStrategy,
        symbols: list[str],
        target_timeframe: str,
        config: PortfolioManagerConfig,
        client: BinanceClient,
        futures_client: BinanceFuturesClient,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        db_path: str | None = None,
        discord_config: DiscordBotConfig | None = None,
        metrics_port: int = 0,
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
    ) -> LiveRunner:
        """Live 모드: LiveDataFeed(Spot) + LiveExecutor(Futures).

        Args:
            client: Spot client (데이터 스트리밍용)
            futures_client: Futures client (주문 실행용)
        """
        feed = LiveDataFeed(symbols, target_timeframe, client)
        executor = LiveExecutor(futures_client)
        runner = cls(
            strategy=strategy,
            feed=feed,
            executor=executor,
            target_timeframe=target_timeframe,
            config=config,
            mode=LiveMode.LIVE,
            initial_capital=initial_capital,
            asset_weights=asset_weights,
            db_path=db_path,
            discord_config=discord_config,
            metrics_port=metrics_port,
            regime_config=regime_config,
            feature_store_config=feature_store_config,
        )
        runner._client = client
        runner._futures_client = futures_client
        runner._symbols = symbols

        # Live 모드에서 DerivativesFeed 자동 생성
        from src.eda.derivatives_feed import LiveDerivativesFeed

        runner._derivatives_feed = LiveDerivativesFeed(symbols, futures_client)
        runner._init_onchain_feed(symbols)
        runner._init_macro_feed()
        runner._init_options_feed()
        runner._init_deriv_ext_feed(symbols)
        return runner

    @classmethod
    def orchestrated_paper(
        cls,
        orchestrator_config: OrchestratorConfig,
        target_timeframe: str,
        client: BinanceClient,
        initial_capital: float = 100_000.0,
        db_path: str | None = None,
        discord_config: DiscordBotConfig | None = None,
        metrics_port: int = 0,
    ) -> LiveRunner:
        """Orchestrated Paper 모드: LiveDataFeed + BacktestExecutor + Orchestrator.

        Args:
            orchestrator_config: OrchestratorConfig
            target_timeframe: 집계 목표 TF
            client: BinanceClient (데이터 스트리밍용)
            initial_capital: 초기 자본
            db_path: SQLite 경로
            discord_config: Discord 설정
            metrics_port: Prometheus 포트

        Returns:
            LiveRunner 인스턴스
        """
        from src.eda.orchestrated_runner import OrchestratedRunner
        from src.orchestrator.allocator import CapitalAllocator
        from src.orchestrator.lifecycle import LifecycleManager
        from src.orchestrator.orchestrator import StrategyOrchestrator
        from src.orchestrator.pod import build_pods
        from src.orchestrator.risk_aggregator import RiskAggregator
        from src.strategy.tsmom.strategy import TSMOMStrategy

        symbols = list(orchestrator_config.all_symbols)
        all_tfs = set(orchestrator_config.all_timeframes)
        multi_tf = len(all_tfs) > 1
        pm_config = OrchestratedRunner.derive_pm_config(orchestrator_config)
        feed = LiveDataFeed(
            symbols,
            target_timeframe,
            client,
            target_timeframes=all_tfs if multi_tf else None,
        )
        executor = BacktestExecutor(cost_model=pm_config.cost_model)

        # Dummy strategy (LiveRunner 생성자 필수)
        dummy_strategy = TSMOMStrategy.from_params()

        runner = cls(
            strategy=dummy_strategy,
            feed=feed,
            executor=executor,
            target_timeframe=target_timeframe,
            config=pm_config,
            mode=LiveMode.PAPER,
            initial_capital=initial_capital,
            asset_weights=dict.fromkeys(symbols, 1.0),
            db_path=db_path,
            discord_config=discord_config,
            metrics_port=metrics_port,
        )
        runner._client = client
        runner._symbols = symbols

        # Orchestrator 생성
        pods = build_pods(orchestrator_config)
        allocator = CapitalAllocator(config=orchestrator_config)
        lifecycle_mgr = LifecycleManager(
            graduation=orchestrator_config.graduation,
            retirement=orchestrator_config.retirement,
        )
        risk_aggregator = RiskAggregator(config=orchestrator_config)

        runner._orchestrator = StrategyOrchestrator(
            config=orchestrator_config,
            pods=pods,
            allocator=allocator,
            lifecycle_manager=lifecycle_mgr,
            risk_aggregator=risk_aggregator,
            target_timeframe=None if multi_tf else target_timeframe,
        )

        return runner

    @classmethod
    def orchestrated_live(
        cls,
        orchestrator_config: OrchestratorConfig,
        target_timeframe: str,
        client: BinanceClient,
        futures_client: BinanceFuturesClient,
        initial_capital: float = 100_000.0,
        db_path: str | None = None,
        discord_config: DiscordBotConfig | None = None,
        metrics_port: int = 0,
    ) -> LiveRunner:
        """Orchestrated Live 모드: LiveDataFeed + LiveExecutor + Orchestrator.

        Args:
            orchestrator_config: OrchestratorConfig
            target_timeframe: 집계 목표 TF
            client: BinanceClient (데이터 스트리밍용)
            futures_client: BinanceFuturesClient (주문 실행용)
            initial_capital: 초기 자본
            db_path: SQLite 경로
            discord_config: Discord 설정
            metrics_port: Prometheus 포트

        Returns:
            LiveRunner 인스턴스
        """
        from src.eda.orchestrated_runner import OrchestratedRunner
        from src.orchestrator.allocator import CapitalAllocator
        from src.orchestrator.lifecycle import LifecycleManager
        from src.orchestrator.orchestrator import StrategyOrchestrator
        from src.orchestrator.pod import build_pods
        from src.orchestrator.risk_aggregator import RiskAggregator
        from src.strategy.tsmom.strategy import TSMOMStrategy

        symbols = list(orchestrator_config.all_symbols)
        all_tfs = set(orchestrator_config.all_timeframes)
        multi_tf = len(all_tfs) > 1
        pm_config = OrchestratedRunner.derive_pm_config(orchestrator_config)
        feed = LiveDataFeed(
            symbols,
            target_timeframe,
            client,
            target_timeframes=all_tfs if multi_tf else None,
        )
        executor = LiveExecutor(futures_client)

        # Dummy strategy
        dummy_strategy = TSMOMStrategy.from_params()

        runner = cls(
            strategy=dummy_strategy,
            feed=feed,
            executor=executor,
            target_timeframe=target_timeframe,
            config=pm_config,
            mode=LiveMode.LIVE,
            initial_capital=initial_capital,
            asset_weights=dict.fromkeys(symbols, 1.0),
            db_path=db_path,
            discord_config=discord_config,
            metrics_port=metrics_port,
        )
        runner._client = client
        runner._futures_client = futures_client
        runner._symbols = symbols

        # Orchestrator 생성
        pods = build_pods(orchestrator_config)
        allocator = CapitalAllocator(config=orchestrator_config)
        lifecycle_mgr = LifecycleManager(
            graduation=orchestrator_config.graduation,
            retirement=orchestrator_config.retirement,
        )
        risk_aggregator = RiskAggregator(config=orchestrator_config)

        runner._orchestrator = StrategyOrchestrator(
            config=orchestrator_config,
            pods=pods,
            allocator=allocator,
            lifecycle_manager=lifecycle_mgr,
            risk_aggregator=risk_aggregator,
            target_timeframe=None if multi_tf else target_timeframe,
        )

        # Live 모드에서 DerivativesFeed 자동 생성
        from src.eda.derivatives_feed import LiveDerivativesFeed

        runner._derivatives_feed = LiveDerivativesFeed(symbols, futures_client)
        return runner

    async def _init_capital_and_db(self) -> tuple[float, Any]:
        """Pre-flight checks + DB 초기화.

        Returns:
            (capital, db) 튜플
        """
        from src.eda.persistence.database import Database

        capital = self._initial_capital
        if self._mode == LiveMode.LIVE and self._futures_client is not None:
            capital = await self._preflight_checks()

        db: Database | None = None
        if self._db_path:
            db = Database(self._db_path)
            await db.connect()

        return capital, db

    async def run(self) -> None:
        """메인 루프. shutdown_event까지 실행."""
        from src.eda.persistence.trade_persistence import TradePersistence

        self._init_tracing()

        # 0. LIVE 모드 Pre-flight checks + DB 초기화
        capital, db = await self._init_capital_and_db()

        discord_tasks: _DiscordTasks | None = None
        pm: EDAPortfolioManager | None = None
        try:
            # 2. 컴포넌트 생성
            bus = EventBus(queue_size=self._queue_size)

            # 2a. 데이터 enrichment providers + StrategyEngine 생성
            (
                regime_service,
                feature_store,
                strategy_engine,
            ) = await self._create_providers_and_engine()
            pm = EDAPortfolioManager(
                config=self._config,
                initial_capital=capital,
                asset_weights=self._asset_weights,
                target_timeframe=self._target_timeframe,
            )
            rm = EDARiskManager(
                config=self._config,
                portfolio_manager=pm,
                max_order_size_usd=capital * self._config.max_leverage_cap,
                enable_circuit_breaker=True,
            )
            # LIVE 모드: 동적 max_order_size 활성화
            if self._mode == LiveMode.LIVE:
                rm.enable_dynamic_max_order_size()
            oms = OMS(executor=self._executor, portfolio_manager=pm)
            analytics = AnalyticsEngine(initial_capital=self._initial_capital)

            # Executor별 초기화
            self._setup_executor(bus, pm)

            # 3. 상태 복구 (DB 활성 시)
            state_mgr = await self._restore_state(db, pm, rm, oms)

            # 3.5. Orchestrator 상태 복구 + daily return MTM 초기화
            orch_persistence = await self._restore_orchestrator_state(state_mgr, capital)

            # 3.6. 거래소 기준 PM reconciliation (sync_capital 전에 실행!)
            await self._reconcile_positions(pm)

            # 3.7. LIVE 모드: state 복원 후 거래소 잔고로 PM/RM 동기화
            if self._mode == LiveMode.LIVE:
                pm.sync_capital(capital)
                rm.sync_peak_equity(capital)
                logger.info(
                    "LIVE sync: PM/RM capital reset to exchange balance ${:.2f}",
                    capital,
                )

            # 3.8. ExchangeStopManager 생성 + 상태 복구 (Live 모드, reconcile 이후)
            exchange_stop_mgr = await self._create_exchange_stop_manager(pm, state_mgr)

            # 4. 모든 컴포넌트 등록 + warmup
            await self._register_and_warmup(
                bus,
                strategy_engine,
                regime_service,
                feature_store,
                pm,
                rm,
                oms,
                analytics,
                exchange_stop_mgr=exchange_stop_mgr,
            )

            # 5. TradePersistence 등록 (analytics 이후)
            if db:
                strategy_name = (
                    "orchestrator" if self._orchestrator is not None else self._strategy.name
                )
                persistence = TradePersistence(db, strategy_name=strategy_name)
                await persistence.register(bus)

            # 5.5. Prometheus MetricsExporter (선택적)
            metrics_exporter = await self._setup_metrics(bus)

            # 6. Discord Bot + NotificationEngine (선택적)
            discord_tasks = await self._setup_discord(
                bus, pm, rm, analytics, exchange_stop_mgr=exchange_stop_mgr
            )

            # 6.5. ExchangeStopManager에 notification_queue 사후 주입
            if exchange_stop_mgr is not None and discord_tasks is not None:
                exchange_stop_mgr._notification_queue = discord_tasks.notification_queue

            # 6.6. LiveOnchainFeed에 notification_queue 사후 주입
            if self._onchain_feed is not None and discord_tasks is not None:
                self._onchain_feed._notification_queue = discord_tasks.notification_queue

            # 7. Signal handler
            self._setup_signal_handlers()

            # 8. 실행
            bus_task = asyncio.create_task(bus.start())
            feed_task = asyncio.create_task(self._feed.start(bus))

            # 주기적 상태 저장 task
            save_task: asyncio.Task[None] | None = None
            if state_mgr:
                save_task = asyncio.create_task(
                    self._periodic_state_save(
                        state_mgr,
                        pm,
                        rm,
                        oms,
                        orch_persistence=orch_persistence,
                        orchestrator=self._orchestrator,
                        exchange_stop_mgr=exchange_stop_mgr,
                    )
                )

            # 주기적 메트릭 갱신 task (uptime + EventBus + exchange health)
            uptime_task: asyncio.Task[None] | None = None
            if metrics_exporter is not None:
                uptime_task = asyncio.create_task(
                    self._periodic_metrics_update(
                        metrics_exporter,
                        bus,
                        self._futures_client,
                        getattr(self, "_orchestrator_metrics", None),
                        ws_detail_callback=getattr(self, "_ws_detail_callback", None),
                        onchain_feed=self._onchain_feed,
                    )
                )

            # 프로세스 모니터 task (event loop lag, RSS, FD)
            process_monitor_task: asyncio.Task[None] | None = None
            if metrics_exporter is not None:
                from src.monitoring.process_monitor import monitor_process_and_loop

                process_monitor_task = asyncio.create_task(
                    monitor_process_and_loop(interval=10.0, bus=bus)
                )

            # Live 모드: PositionReconciler 초기 + 주기적 검증
            notification_queue = discord_tasks.notification_queue if discord_tasks else None
            reconciler_task = await self._setup_reconciler(pm, rm, notification_queue)

            # 구조화된 startup summary
            self._log_startup_summary(capital)

            # Lifecycle: startup 알림 (Discord Bot ready 대기 포함)
            await self._send_lifecycle_startup(discord_tasks, capital)

            logger.info("LiveRunner started (mode={}, db={})", self._mode.value, self._db_path)

            # 8. Shutdown 대기
            await self._shutdown_event.wait()

            # 9. Graceful shutdown
            await self._graceful_shutdown(
                bus=bus,
                bus_task=bus_task,
                feed_task=feed_task,
                save_task=save_task,
                uptime_task=uptime_task,
                process_monitor_task=process_monitor_task,
                reconciler_task=reconciler_task,
                pm=pm,
                rm=rm,
                oms=oms,
                analytics=analytics,
                discord_tasks=discord_tasks,
                state_mgr=state_mgr,
                orch_persistence=orch_persistence,
                exchange_stop_mgr=exchange_stop_mgr,
            )
        except Exception as exc:
            # Prometheus errors_counter (lazy import)
            try:
                from src.monitoring.metrics import errors_counter

                errors_counter.labels(
                    component="LiveRunner", error_type=type(exc).__name__
                ).inc()
            except Exception:  # noqa: S110
                pass
            await self._send_lifecycle_crash(discord_tasks, exc, pm=pm)
            raise
        finally:
            if db:
                await db.close()

    async def _graceful_shutdown(
        self,
        *,
        bus: EventBus,
        bus_task: asyncio.Task[None],
        feed_task: asyncio.Task[None],
        save_task: asyncio.Task[None] | None,
        uptime_task: asyncio.Task[None] | None,
        process_monitor_task: asyncio.Task[None] | None,
        reconciler_task: asyncio.Task[None] | None,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        oms: OMS,
        analytics: AnalyticsEngine,
        discord_tasks: _DiscordTasks | None,
        state_mgr: Any,
        orch_persistence: Any,
        exchange_stop_mgr: Any = None,
    ) -> None:
        """Graceful shutdown 시퀀스 — run()에서 분리."""
        logger.warning("Initiating graceful shutdown...")

        # 주기적 task 취소
        await self._cancel_periodic_tasks(
            save_task, uptime_task, process_monitor_task, reconciler_task
        )

        await self._feed.stop()
        feed_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await feed_task

        # DerivativesFeed / OnchainFeed / Macro / Options / DerivExt 종료
        await self._stop_derivatives_feed()
        await self._stop_onchain_feed()
        await self._stop_macro_feed()
        await self._stop_options_feed()
        await self._stop_deriv_ext_feed()

        if self._orchestrator is not None:
            await self._orchestrator.flush_pending_signals()
            self._orchestrator.flush_daily_returns()
        await pm.flush_pending_signals()
        await bus.flush()
        await bus.stop()
        await bus_task

        # Exchange Safety Stops: shutdown 처리
        if exchange_stop_mgr is not None:
            if self._config.cancel_stops_on_shutdown:
                await exchange_stop_mgr.cancel_all_stops()
                logger.info("Exchange safety stops cancelled on shutdown")
            else:
                active = exchange_stop_mgr.active_stops
                if active:
                    logger.info(
                        "Exchange safety stops RETAINED on shutdown ({} symbols) — continues until restart",
                        len(active),
                    )

        # Lifecycle: shutdown 알림
        await self._send_lifecycle_shutdown(discord_tasks, pm, analytics)

        # Discord Bot/Queue 종료
        await self._shutdown_discord(discord_tasks)

        # 최종 상태 저장
        if state_mgr:
            exchange_stops_state = (
                exchange_stop_mgr.get_state() if exchange_stop_mgr is not None else None
            )
            await state_mgr.save_all(pm, rm, oms=oms, exchange_stops_state=exchange_stops_state)
            if orch_persistence is not None and self._orchestrator is not None:
                await orch_persistence.save(self._orchestrator)
            logger.info("Final state saved")

        # OTel tracing 종료
        shutdown_tracing()

        logger.info("LiveRunner stopped gracefully")

    @staticmethod
    async def _cancel_periodic_tasks(*tasks: asyncio.Task[None] | None) -> None:
        """주기적 tasks를 안전하게 취소."""
        for task in tasks:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    def request_shutdown(self) -> None:
        """외부에서 shutdown 요청 (테스트용)."""
        self._shutdown_event.set()

    @staticmethod
    def _init_tracing() -> None:
        """OTel tracing 초기화 (MC_OTEL_ENDPOINT 환경변수 설정 시)."""
        import os

        otel_endpoint = os.environ.get("MC_OTEL_ENDPOINT")
        if otel_endpoint:
            setup_tracing(endpoint=otel_endpoint)

    async def _restore_orchestrator_state(self, state_mgr: Any, capital: float = 0.0) -> Any:
        """Orchestrator 상태를 복구하고 daily return MTM을 초기화합니다.

        Args:
            state_mgr: StateManager 또는 None
            capital: 초기 자본 (daily return MTM 초기화용)

        Returns:
            OrchestratorStatePersistence 또는 None
        """
        if self._orchestrator is None:
            return None

        # Daily return MTM 초기화 (set_base_equity는 복원된 값을 보호)
        if capital > 0:
            self._orchestrator.set_initial_capital(capital)

        if state_mgr is None:
            return None

        from src.orchestrator.state_persistence import OrchestratorStatePersistence

        orch_persistence = OrchestratorStatePersistence(state_mgr)
        restored = await orch_persistence.restore(self._orchestrator)
        if restored:
            logger.info("Orchestrator state restored")
        return orch_persistence

    async def _create_exchange_stop_manager(
        self,
        pm: EDAPortfolioManager,
        state_mgr: Any,
    ) -> Any:
        """ExchangeStopManager 생성 + 상태 복구 (Live 모드 전용).

        Returns:
            ExchangeStopManager 또는 None
        """
        if (
            self._mode != LiveMode.LIVE
            or self._futures_client is None
            or not self._config.use_exchange_safety_stop
        ):
            return None

        from src.eda.exchange_stop_manager import ExchangeStopManager

        mgr = ExchangeStopManager(self._config, self._futures_client, pm)

        # 상태 복구
        if state_mgr is not None:
            stops_state = await state_mgr.load_exchange_stops_state()
            if stops_state:
                mgr.restore_state(stops_state)

        # 거래소 실제 주문 존재 여부 검증 (notification_queue는 사후 주입)
        await mgr.verify_exchange_stops()

        # 포지션 있는데 stop 없는 심볼에 안전망 stop 재배치
        placed = await mgr.place_missing_stops()
        if placed:
            logger.info("Safety stops re-placed for {} symbols", placed)

        return mgr

    @staticmethod
    async def _restore_state(
        db: Database | None,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        oms: OMS | None = None,
    ) -> StateManager | None:
        """DB에서 PM/RM/OMS 상태를 복구.

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
        if oms is not None:
            oms_state = await state_mgr.load_oms_state()
            if oms_state:
                oms.restore_processed_orders(oms_state)
                logger.info("OMS state restored ({} orders)", len(oms_state))
        return state_mgr

    async def _setup_discord(
        self,
        bus: EventBus,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        analytics: AnalyticsEngine,
        exchange_stop_mgr: Any = None,
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
            orchestrator=self._orchestrator,
            exchange_stop_mgr=exchange_stop_mgr,
            onchain_feed=self._onchain_feed,
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

        # HealthCheckScheduler 생성 + 시작
        from src.notification.health_scheduler import HealthCheckScheduler

        health_scheduler = HealthCheckScheduler(
            queue=notification_queue,
            pm=pm,
            rm=rm,
            analytics=analytics,
            feed=self._feed,
            bus=bus,
            futures_client=self._futures_client,
            symbols=self._symbols,
            exchange_stop_mgr=exchange_stop_mgr,
            onchain_feed=self._onchain_feed,
        )
        await health_scheduler.start()

        # L3-3: TradingContext에 trigger 콜백 설정
        trading_ctx.report_trigger = report_scheduler.trigger_daily_report
        trading_ctx.health_trigger = health_scheduler.trigger_health_check

        # OrchestratorNotificationEngine (Orchestrator 활성 시)
        if self._orchestrator is not None:
            from src.notification.orchestrator_engine import OrchestratorNotificationEngine

            orch_notification = OrchestratorNotificationEngine(
                notification_queue, self._orchestrator
            )
            self._orchestrator.set_notification_engine(orch_notification)
            await orch_notification.start()

        logger.info(
            "Discord Bot + NotificationEngine + ReportScheduler + HealthCheckScheduler enabled"
        )

        return _DiscordTasks(
            bot_service=bot_service,
            notification_queue=notification_queue,
            queue_task=queue_task,
            bot_task=bot_task,
            report_scheduler=report_scheduler,
            health_scheduler=health_scheduler,
        )

    @staticmethod
    async def _wait_for_discord_ready(
        bot_service: DiscordBotService,
        max_wait: float = 15.0,
    ) -> None:
        """Discord Bot이 ready 상태가 될 때까지 대기.

        Args:
            bot_service: DiscordBotService 인스턴스
            max_wait: 최대 대기 시간 (초)
        """
        _poll_interval = 0.2
        elapsed = 0.0
        while not bot_service.is_ready and elapsed < max_wait:
            await asyncio.sleep(_poll_interval)
            elapsed += _poll_interval
        if not bot_service.is_ready:
            logger.warning("Discord Bot not ready after {:.1f}s, proceeding anyway", max_wait)

    @staticmethod
    async def _shutdown_discord(tasks: _DiscordTasks | None) -> None:
        """Discord Bot/Queue/ReportScheduler/HealthCheckScheduler 정리."""
        if tasks is None:
            return
        if tasks.health_scheduler is not None:
            await tasks.health_scheduler.stop()
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

    async def _create_providers_and_engine(
        self,
    ) -> tuple[RegimeService | None, FeatureStore | None, StrategyEngine | None]:
        """데이터 enrichment providers + StrategyEngine 생성.

        Returns:
            (regime_service, feature_store, strategy_engine) 튜플
        """
        derivatives_provider = await self._start_derivatives_feed()
        regime_service = self._create_regime_service(derivatives_provider)
        onchain_provider = await self._start_onchain_feed()
        feature_store = self._create_feature_store()
        macro_provider = await self._start_macro_feed()
        options_provider = await self._start_options_feed()
        deriv_ext_provider = await self._start_deriv_ext_feed()

        strategy_engine: StrategyEngine | None = None
        if self._orchestrator is None:
            strategy_engine = StrategyEngine(
                self._strategy,
                target_timeframe=self._target_timeframe,
                regime_service=regime_service,
                derivatives_provider=derivatives_provider,
                feature_store=feature_store,
                onchain_provider=onchain_provider,
                macro_provider=macro_provider,
                options_provider=options_provider,
                deriv_ext_provider=deriv_ext_provider,
            )
        return regime_service, feature_store, strategy_engine

    def _create_regime_service(
        self,
        derivatives_provider: object | None = None,
    ) -> RegimeService | None:
        """RegimeService 생성 (regime_config가 None이면 None).

        Args:
            derivatives_provider: 파생상품 데이터 프로바이더 (있으면 DerivativesDetector 활성화)
        """
        if self._regime_config is None:
            return None

        from src.regime.service import RegimeService

        return RegimeService(self._regime_config, derivatives_provider=derivatives_provider)

    def _create_feature_store(self) -> FeatureStore | None:
        """FeatureStore 생성 (feature_store_config가 None이면 None)."""
        if self._feature_store_config is None:
            return None

        from src.market.feature_store import FeatureStore

        return FeatureStore(self._feature_store_config)

    async def _warmup_strategy(
        self,
        strategy_engine: StrategyEngine,
        *,
        regime_service: RegimeService | None = None,
        feature_store: FeatureStore | None = None,
    ) -> None:
        """REST API로 과거 데이터를 가져와 StrategyEngine 버퍼에 주입.

        _client가 None이면 스킵. 심볼별 독립 실행으로 1개 실패해도 나머지 진행.
        regime_service가 있으면 warmup bars로 regime detector도 초기화합니다.
        """
        if self._client is None:
            logger.warning("No client available, skipping REST API warmup")
            return

        warmup_needed = strategy_engine.warmup_periods + 10  # 여유분
        symbols = self._feed.symbols

        logger.info(
            "Starting REST API warmup: {} symbols, {} bars each",
            len(symbols),
            warmup_needed,
        )

        for symbol in symbols:
            try:
                bars, timestamps = await self._fetch_warmup_bars(
                    symbol, self._target_timeframe, warmup_needed
                )
                if bars:
                    strategy_engine.inject_warmup(symbol, bars, timestamps)
                    # RegimeService warmup
                    if regime_service is not None:
                        closes = [bar["close"] for bar in bars]
                        regime_service.warmup(symbol, closes)
                    # FeatureStore warmup
                    if feature_store is not None:
                        import pandas as pd

                        warmup_df = pd.DataFrame(bars, index=pd.DatetimeIndex(timestamps))
                        feature_store.warmup(symbol, warmup_df)
                else:
                    logger.warning("No warmup data fetched for {}", symbol)
            except Exception:
                logger.exception("Warmup failed for {}, continuing without warmup", symbol)

    async def _warmup_orchestrator(self, orchestrator: StrategyOrchestrator) -> None:
        """REST API로 과거 데이터를 가져와 Orchestrator의 각 Pod 버퍼에 주입.

        _client가 None이면 스킵. Pod별·심볼별 독립 실행으로 1개 실패해도 나머지 진행.
        """
        if self._client is None:
            logger.warning("No client available, skipping orchestrator warmup")
            return

        for pod in orchestrator.pods:
            warmup_needed = pod.warmup_periods + 10  # 여유분
            for symbol in pod.symbols:
                try:
                    bars, timestamps = await self._fetch_warmup_bars(
                        symbol, pod.timeframe, warmup_needed
                    )
                    if bars:
                        pod.inject_warmup(symbol, bars, timestamps)
                        logger.debug(
                            "Warmup injected: pod={}, symbol={}, {} bars",
                            pod.pod_id,
                            symbol,
                            len(bars),
                        )
                    else:
                        logger.warning(
                            "No warmup data for pod={}, symbol={}",
                            pod.pod_id,
                            symbol,
                        )
                except Exception:
                    logger.exception(
                        "Warmup failed for pod={}, symbol={}, continuing",
                        pod.pod_id,
                        symbol,
                    )

        logger.info(
            "Orchestrator warmup complete: {} pods",
            len(orchestrator.pods),
        )

    async def _fetch_warmup_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> tuple[list[dict[str, float]], list[datetime]]:
        """REST API로 과거 OHLCV 데이터를 가져와 StrategyEngine 버퍼 포맷으로 변환.

        마지막 캔들은 미완성 가능성이 있으므로 제거합니다.

        Args:
            symbol: 거래 심볼
            timeframe: 타임프레임 (e.g. "1D", "4h")
            limit: 가져올 캔들 수

        Returns:
            (bars, timestamps) 튜플
        """
        assert self._client is not None

        # CCXT TF 변환: Binance는 소문자만 지원 (1D → 1d, 12H → 12h)
        ccxt_tf = timeframe.lower()

        raw: list[list[Any]] = await self._client.fetch_ohlcv_raw(
            symbol, ccxt_tf, limit=min(limit, 1000)
        )

        if len(raw) < _MIN_WARMUP_CANDLES:
            return [], []

        # 마지막 캔들 제거 (미완성 가능성)
        raw = raw[:-1]

        bars: list[dict[str, float]] = []
        timestamps: list[datetime] = []

        for candle in raw:
            ts_ms = int(candle[0])
            bars.append(
                {
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                }
            )
            timestamps.append(datetime.fromtimestamp(ts_ms / 1000, tz=UTC))

        logger.info("Fetched {} warmup bars for {} (requested {})", len(bars), symbol, limit)
        return bars, timestamps

    def _setup_executor(self, bus: EventBus, pm: EDAPortfolioManager) -> None:
        """Executor별 초기화.

        - BacktestExecutor: BAR 핸들러 등록
        - LiveExecutor: PM 참조 설정
        """
        if isinstance(self._executor, BacktestExecutor):
            bt_executor = self._executor

            async def executor_bar_handler(event: AnyEvent) -> None:
                assert isinstance(event, BarEvent)
                bt_executor.on_bar(event)

            bus.subscribe(EventType.BAR, executor_bar_handler)
        elif isinstance(self._executor, LiveExecutor):
            self._executor.set_pm(pm)

    async def _register_and_warmup(
        self,
        bus: EventBus,
        strategy_engine: StrategyEngine | None,
        regime_service: RegimeService | None,
        feature_store: FeatureStore | None,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        oms: OMS,
        analytics: AnalyticsEngine,
        exchange_stop_mgr: Any = None,
    ) -> None:
        """모든 컴포넌트 등록 + REST API warmup."""
        # RegimeService → FeatureStore → StrategyEngine/Orchestrator 순서
        if regime_service is not None:
            await regime_service.register(bus)
        if feature_store is not None:
            await feature_store.register(bus)

        if self._orchestrator is not None:
            await self._orchestrator.register(bus)
        elif strategy_engine is not None:
            await strategy_engine.register(bus)

        await pm.register(bus)
        await rm.register(bus)
        await oms.register(bus)
        await analytics.register(bus)

        # ExchangeStopManager: PM/OMS 이후 등록 (fill 처리 후 stop 관리)
        if exchange_stop_mgr is not None:
            await exchange_stop_mgr.register(bus)

        # REST API warmup — WebSocket 시작 전 버퍼 사전 채움
        if self._orchestrator is not None:
            await self._warmup_orchestrator(self._orchestrator)
            # 검출기 자동 초기화 (warmup 완료 후)
            lifecycle = self._orchestrator.lifecycle
            if lifecycle is not None:
                for pod in self._orchestrator.pods:
                    if pod.daily_returns:
                        lifecycle.auto_init_detectors(pod.pod_id, list(pod.daily_returns))
        elif strategy_engine is not None:
            await self._warmup_strategy(
                strategy_engine, regime_service=regime_service, feature_store=feature_store
            )

    async def _reconcile_positions(self, pm: EDAPortfolioManager) -> list[str]:
        """거래소 포지션 기준으로 PM phantom position 제거.

        LIVE 모드 + futures_client 존재 시에만 동작합니다.
        API 실패 시 빈 리스트 반환 (safety-first, 기존 동작 유지).

        Returns:
            제거된 심볼 리스트
        """
        if self._mode != LiveMode.LIVE or self._futures_client is None:
            return []

        from src.eda.reconciler import PositionReconciler

        try:
            exchange_positions = await PositionReconciler.parse_exchange_positions(
                self._futures_client, self._symbols
            )
            removed = pm.reconcile_with_exchange(exchange_positions)
        except Exception:
            logger.exception("Startup reconciliation failed — continuing with existing state")
            return []
        else:
            if removed:
                logger.warning(
                    "Startup reconciliation: removed {} phantom positions: {}",
                    len(removed),
                    removed,
                )
            return removed

    async def _setup_reconciler(
        self,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        notification_queue: NotificationQueue | None = None,
    ) -> asyncio.Task[None] | None:
        """Live 모드: PositionReconciler 초기 + 주기적 검증 task 생성."""
        if self._mode != LiveMode.LIVE or self._futures_client is None:
            return None

        from src.eda.reconciler import PositionReconciler

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, self._futures_client, self._symbols)

        # Startup drift → Discord 알림
        if drifts and notification_queue is not None:
            from src.notification.reconciler_formatters import format_position_drift_embed

            details = reconciler.last_drift_details
            if details:
                embed = format_position_drift_embed(details)
                await notification_queue.enqueue(
                    NotificationItem(
                        severity=Severity.WARNING,
                        channel=ChannelRoute.ALERTS,
                        embed=embed,
                        spam_key="startup_drift",
                    )
                )

        return asyncio.create_task(
            self._periodic_reconciliation(
                reconciler,
                pm,
                rm,
                self._futures_client,
                self._symbols,
                notification_queue=notification_queue,
            )
        )

    async def _stop_derivatives_feed(self) -> None:
        """DerivativesFeed 종료 (설정된 경우)."""
        if self._derivatives_feed is not None:
            await self._derivatives_feed.stop()

    async def _start_derivatives_feed(self) -> Any:
        """DerivativesFeed 시작 (설정된 경우).

        Returns:
            DerivativesProvider 또는 None
        """
        if self._derivatives_feed is None:
            return None
        await self._derivatives_feed.start()
        return self._derivatives_feed

    def _init_onchain_feed(self, symbols: list[str]) -> None:
        """LiveOnchainFeed 인스턴스 생성 (Silver 데이터 존재 시)."""
        from src.eda.onchain_feed import LiveOnchainFeed

        self._onchain_feed = LiveOnchainFeed(symbols)

    async def _start_onchain_feed(self) -> Any:
        """OnchainFeed 시작 (설정된 경우).

        Returns:
            OnchainProvider 또는 None
        """
        if self._onchain_feed is None:
            return None
        await self._onchain_feed.start()
        # 캐시가 비어있으면 None 반환 (Silver 데이터 없음)
        if not self._onchain_feed._cache:
            await self._onchain_feed.stop()
            self._onchain_feed = None
            return None
        return self._onchain_feed

    async def _stop_onchain_feed(self) -> None:
        """OnchainFeed 종료 (설정된 경우)."""
        if self._onchain_feed is not None:
            await self._onchain_feed.stop()

    def _init_macro_feed(self) -> None:
        """LiveMacroFeed 인스턴스 생성."""
        from src.eda.macro_feed import LiveMacroFeed

        self._macro_feed = LiveMacroFeed()

    async def _start_macro_feed(self) -> Any:
        """MacroFeed 시작 (설정된 경우).

        Returns:
            MacroProvider 또는 None
        """
        if self._macro_feed is None:
            return None
        await self._macro_feed.start()
        if not self._macro_feed._cache:
            await self._macro_feed.stop()
            self._macro_feed = None
            return None
        return self._macro_feed

    async def _stop_macro_feed(self) -> None:
        """MacroFeed 종료 (설정된 경우)."""
        if self._macro_feed is not None:
            await self._macro_feed.stop()

    def _init_options_feed(self) -> None:
        """LiveOptionsFeed 인스턴스 생성."""
        from src.eda.options_feed import LiveOptionsFeed

        self._options_feed = LiveOptionsFeed()

    async def _start_options_feed(self) -> Any:
        """OptionsFeed 시작 (설정된 경우).

        Returns:
            OptionsProvider 또는 None
        """
        if self._options_feed is None:
            return None
        await self._options_feed.start()
        if not self._options_feed._cache:
            await self._options_feed.stop()
            self._options_feed = None
            return None
        return self._options_feed

    async def _stop_options_feed(self) -> None:
        """OptionsFeed 종료 (설정된 경우)."""
        if self._options_feed is not None:
            await self._options_feed.stop()

    def _init_deriv_ext_feed(self, symbols: list[str]) -> None:
        """LiveDerivExtFeed 인스턴스 생성."""
        from src.eda.deriv_ext_feed import LiveDerivExtFeed

        self._deriv_ext_feed = LiveDerivExtFeed(symbols)

    async def _start_deriv_ext_feed(self) -> Any:
        """DerivExtFeed 시작 (설정된 경우).

        Returns:
            DerivExtProvider 또는 None
        """
        if self._deriv_ext_feed is None:
            return None
        await self._deriv_ext_feed.start()
        if not self._deriv_ext_feed._cache:
            await self._deriv_ext_feed.stop()
            self._deriv_ext_feed = None
            return None
        return self._deriv_ext_feed

    async def _stop_deriv_ext_feed(self) -> None:
        """DerivExtFeed 종료 (설정된 경우)."""
        if self._deriv_ext_feed is not None:
            await self._deriv_ext_feed.stop()

    async def _preflight_checks(self) -> float:
        """LIVE 모드 시작 전 거래소 상태 검증.

        - USDT 잔고 조회 + 0 이하 검증
        - config capital과 50% 이상 차이 시 WARNING
        - 미체결 주문 감지 시 CRITICAL 로그

        Returns:
            거래소 USDT 잔고 (PM initial_capital로 사용)

        Raises:
            RuntimeError: 잔고 0 이하
        """
        assert self._futures_client is not None
        logger.info("Running pre-flight checks...")

        # 1. 잔고 확인
        balance = await self._futures_client.fetch_balance()
        usdt_info = balance.get("USDT", {})
        total_balance = float(usdt_info.get("total", 0) if isinstance(usdt_info, dict) else 0)

        if total_balance <= 0:
            msg = f"Pre-flight FAIL: USDT balance is {total_balance:.2f} (must be > 0)"
            logger.critical(msg)
            raise RuntimeError(msg)

        logger.info("Pre-flight: USDT balance = ${:.2f}", total_balance)

        # 2. Config capital 대비 차이 확인
        config_capital = self._initial_capital
        if config_capital > 0:
            _balance_warn_ratio = 0.5
            diff_ratio = abs(total_balance - config_capital) / config_capital
            if diff_ratio > _balance_warn_ratio:
                logger.warning(
                    "Pre-flight WARNING: exchange balance ${:.0f} differs from config capital ${:.0f} by {:.0%}",
                    total_balance,
                    config_capital,
                    diff_ratio,
                )

        # 3. 미체결 주문 감지 (safety-stop 주문 제외)
        from src.eda.exchange_stop_manager import SAFETY_STOP_PREFIX

        for symbol in self._symbols:
            futures_symbol = self._futures_client.to_futures_symbol(symbol)
            try:
                open_orders = await self._futures_client.fetch_open_orders(futures_symbol)
                # safety-stop prefix 주문은 제외 (이전 세션의 안전망)
                non_safety = [
                    o
                    for o in open_orders
                    if not str(o.get("clientOrderId", "")).startswith(SAFETY_STOP_PREFIX)
                ]
                if non_safety:
                    logger.critical(
                        "Pre-flight: {} stale open orders for {} — cancel manually before trading!",
                        len(non_safety),
                        symbol,
                    )
                elif open_orders:
                    logger.info(
                        "Pre-flight: {} safety-stop orders found for {} (retained from previous session)",
                        len(open_orders),
                        symbol,
                    )
            except Exception:
                logger.warning("Pre-flight: Failed to check open orders for {}", symbol)

        logger.info("Pre-flight checks PASSED — using exchange balance ${:.2f}", total_balance)
        return total_balance

    def _log_startup_summary(self, capital: float) -> None:
        """구조화된 startup summary 로그."""
        strategy_label = (
            f"Orchestrator ({len(self._orchestrator.pods)} pods)"
            if self._orchestrator is not None
            else self._strategy.name
        )
        summary = (
            f"=== LiveRunner Startup Summary ===\n"
            f"  Mode:       {self._mode.value}\n"
            f"  Strategy:   {strategy_label}\n"
            f"  Symbols:    {', '.join(self._symbols) if self._symbols else 'N/A'}\n"
            f"  Timeframe:  {self._target_timeframe}\n"
            f"  Capital:    ${capital:,.2f}\n"
            f"  Leverage:   {self._config.max_leverage_cap:.1f}x\n"
            f"  SL:         {f'{self._config.system_stop_loss:.0%}' if self._config.system_stop_loss else 'OFF'}\n"
            f"  TS:         {f'{self._config.trailing_stop_atr_multiplier:.1f}x ATR' if self._config.use_trailing_stop else 'OFF'}"
        )
        logger.info("{}", summary)

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
        oms: OMS | None = None,
        orch_persistence: Any = None,
        orchestrator: Any = None,
        exchange_stop_mgr: Any = None,
        interval: float = _DEFAULT_SAVE_INTERVAL,
    ) -> None:
        """주기적으로 PM/RM/OMS/ExchangeStops/Orchestrator 상태를 저장."""
        while True:
            await asyncio.sleep(interval)
            try:
                exchange_stops_state = (
                    exchange_stop_mgr.get_state() if exchange_stop_mgr is not None else None
                )
                await state_mgr.save_all(pm, rm, oms=oms, exchange_stops_state=exchange_stops_state)
                if orch_persistence is not None and orchestrator is not None:
                    await orch_persistence.save(orchestrator)
            except Exception:
                logger.exception("Periodic state save failed")

    async def _setup_metrics(self, bus: EventBus) -> MetricsExporter | None:
        """Prometheus MetricsExporter 초기화 (선택적).

        bot_info, trading_mode_enum을 설정합니다.

        Returns:
            MetricsExporter 또는 None (비활성 시)
        """
        if self._metrics_port <= 0:
            return None

        from src.monitoring.metrics import (
            MetricsExporter,
            PrometheusApiCallback,
            PrometheusWsDetailCallback,
            bot_info,
            trading_mode_enum,
        )

        exporter = MetricsExporter(port=self._metrics_port)
        await exporter.register(bus)
        exporter.start_server()

        # Meta 메트릭 설정
        bot_info.info(
            {
                "version": "0.1.0",
                "mode": self._mode.value,
                "exchange": "binance",
                "strategy": self._strategy.name,
            }
        )
        trading_mode_enum.state(self._mode.value)

        # LIVE 모드: BinanceFuturesClient에 PrometheusApiCallback 주입
        # + LiveExecutor에 PrometheusLiveExecutorMetrics 주입
        if self._futures_client is not None:
            self._futures_client.set_metrics_callback(PrometheusApiCallback())

        if isinstance(self._executor, LiveExecutor):
            from src.monitoring.metrics import PrometheusLiveExecutorMetrics

            self._executor.set_metrics(PrometheusLiveExecutorMetrics())

        # LiveDataFeed에 WS 상태 콜백 주입 (상세 메트릭 포함)
        ws_detail_cb = PrometheusWsDetailCallback()
        self._feed.set_ws_callback(ws_detail_cb)
        self._ws_detail_callback: PrometheusWsDetailCallback | None = ws_detail_cb

        # OrchestratorMetrics (Orchestrator 활성 시)
        if self._orchestrator is not None:
            from src.orchestrator.metrics import OrchestratorMetrics

            self._orchestrator_metrics: OrchestratorMetrics | None = OrchestratorMetrics(
                self._orchestrator
            )
        else:
            self._orchestrator_metrics = None

        return exporter

    @staticmethod
    async def _periodic_metrics_update(
        exporter: MetricsExporter,
        bus: EventBus,
        futures_client: BinanceFuturesClient | None,
        orchestrator_metrics: OrchestratorMetrics | None = None,
        ws_detail_callback: Any = None,
        onchain_feed: Any = None,
        interval: float = 30.0,
    ) -> None:
        """주기적으로 uptime + EventBus + exchange health + bar ages + orchestrator + onchain 메트릭 갱신."""
        while True:
            await asyncio.sleep(interval)
            exporter.update_uptime()
            exporter.update_eventbus_metrics(bus)
            exporter.update_bar_ages()
            if futures_client is not None:
                exporter.update_exchange_health(futures_client.consecutive_failures)
            if orchestrator_metrics is not None:
                orchestrator_metrics.update()
            if ws_detail_callback is not None:
                ws_detail_callback.update_message_ages()
            if onchain_feed is not None:
                onchain_feed.update_cache_metrics()
            # Execution health check — fill rate gauge + alert
            from src.monitoring.metrics import execution_fill_rate_gauge

            fill_rate_anomaly = exporter.check_execution_health()
            if fill_rate_anomaly is not None:
                execution_fill_rate_gauge.set(fill_rate_anomaly.current_value)
                await exporter.publish_execution_alert(
                    fill_rate_anomaly.severity, fill_rate_anomaly.message
                )
            else:
                execution_fill_rate_gauge.set(exporter.get_execution_fill_rate())
            # HeartbeatEvent 발행 — MetricsExporter가 heartbeat_timestamp gauge 갱신
            await bus.publish(HeartbeatEvent(component="LiveRunner"))

    @staticmethod
    async def _periodic_reconciliation(
        reconciler: PositionReconciler,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        futures_client: BinanceFuturesClient,
        symbols: list[str],
        interval: float = _RECONCILER_INTERVAL,
        notification_queue: NotificationQueue | None = None,
    ) -> None:
        """주기적 포지션 + 잔고 교차 검증."""
        from src.notification.reconciler_formatters import (
            format_balance_drift_embed,
            format_position_drift_embed,
        )

        # Balance drift 알림 임계값 (%)
        _balance_notify_threshold = 2.0

        while True:
            await asyncio.sleep(interval)
            drifts = await reconciler.periodic_check(pm, futures_client, symbols)

            # Position drift 알림
            if drifts and notification_queue is not None:
                details = reconciler.last_drift_details
                if details:
                    embed = format_position_drift_embed(details)
                    await notification_queue.enqueue(
                        NotificationItem(
                            severity=Severity.WARNING,
                            channel=ChannelRoute.ALERTS,
                            embed=embed,
                            spam_key="position_drift",
                        )
                    )

            # 잔고 검증 + RM equity sync
            exchange_equity = await reconciler.check_balance(pm, futures_client)
            if exchange_equity is not None:
                balance_drift = reconciler.last_balance_drift_pct

                # Balance drift가 작을 때만 RM peak sync
                # 입금/출금 시 peak 오염 → 거짓 CircuitBreaker 방지
                if balance_drift < _balance_notify_threshold:
                    await rm.sync_exchange_equity(exchange_equity)
                else:
                    logger.info(
                        "RM peak sync skipped: balance drift {:.1f}% (deposit/withdrawal suspected)",
                        balance_drift,
                    )

                # Balance drift 알림
                if balance_drift >= _balance_notify_threshold and notification_queue is not None:
                    embed = format_balance_drift_embed(
                        pm_equity=pm.total_equity,
                        exchange_equity=exchange_equity,
                        drift_pct=balance_drift,
                    )
                    await notification_queue.enqueue(
                        NotificationItem(
                            severity=Severity.WARNING,
                            channel=ChannelRoute.ALERTS,
                            embed=embed,
                            spam_key="balance_drift",
                        )
                    )

    @property
    def feed(self) -> LiveDataFeed:
        """LiveDataFeed 인스턴스."""
        return self._feed

    @property
    def mode(self) -> LiveMode:
        """실행 모드."""
        return self._mode

    # ─── Lifecycle Alerts ─────────────────────────────────────

    async def _send_lifecycle_startup(
        self,
        discord_tasks: _DiscordTasks | None,
        capital: float,
    ) -> None:
        """봇 시작 알림을 Discord ALERTS 채널에 전송.

        Discord Bot이 ready 상태가 될 때까지 대기한 후 전송합니다.
        """
        if discord_tasks is None:
            return

        await self._wait_for_discord_ready(discord_tasks.bot_service)

        from src.notification.lifecycle import format_startup_embed

        strategy_label = (
            f"Orchestrator ({len(self._orchestrator.pods)} pods)"
            if self._orchestrator is not None
            else self._strategy.name
        )
        pod_summaries = (
            self._orchestrator.get_pod_summary() if self._orchestrator is not None else None
        )
        embed = format_startup_embed(
            mode=self._mode.value,
            strategy_name=strategy_label,
            symbols=self._symbols,
            capital=capital,
            timeframe=self._target_timeframe,
            pod_summaries=pod_summaries,
        )
        item = NotificationItem(
            severity=Severity.INFO,
            channel=ChannelRoute.ALERTS,
            embed=embed,
        )
        await discord_tasks.notification_queue.enqueue(item)

    async def _send_lifecycle_shutdown(
        self,
        discord_tasks: _DiscordTasks | None,
        pm: EDAPortfolioManager,
        analytics: AnalyticsEngine,
    ) -> None:
        """봇 정상 종료 알림을 Discord ALERTS 채널에 전송."""
        if discord_tasks is None:
            return

        from src.notification.lifecycle import format_shutdown_embed

        uptime = time.monotonic() - self._start_time
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        trades = analytics.closed_trades
        realized_pnl = sum(
            float(t.pnl)
            for t in trades
            if t.exit_time and t.exit_time >= today_start and t.pnl is not None
        )
        unrealized_pnl = sum(p.unrealized_pnl for p in pm.positions.values() if p.is_open)
        pod_summaries = (
            self._orchestrator.get_pod_summary() if self._orchestrator is not None else None
        )

        embed = format_shutdown_embed(
            reason="Graceful shutdown",
            uptime_seconds=uptime,
            final_equity=pm.total_equity,
            initial_capital=pm.initial_capital,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            open_positions=pm.open_position_count,
            pod_summaries=pod_summaries,
        )
        item = NotificationItem(
            severity=Severity.WARNING,
            channel=ChannelRoute.ALERTS,
            embed=embed,
        )
        await discord_tasks.notification_queue.enqueue(item)

    async def _send_lifecycle_crash(
        self,
        discord_tasks: _DiscordTasks | None,
        exc: Exception,
        pm: EDAPortfolioManager | None = None,
    ) -> None:
        """봇 비정상 종료 알림을 Discord ALERTS 채널에 전송."""
        if discord_tasks is None:
            return

        from src.notification.lifecycle import format_crash_embed

        uptime = time.monotonic() - self._start_time

        final_equity: float | None = None
        open_positions: int | None = None
        unrealized_pnl: float | None = None
        if pm is not None:
            try:
                final_equity = pm.total_equity
                open_positions = pm.open_position_count
                unrealized_pnl = sum(p.unrealized_pnl for p in pm.positions.values() if p.is_open)
            except Exception:
                logger.debug("Could not collect PM state for crash embed")

        embed = format_crash_embed(
            error_type=type(exc).__name__,
            error_message=str(exc),
            uptime_seconds=uptime,
            final_equity=final_equity,
            open_positions=open_positions,
            unrealized_pnl=unrealized_pnl,
        )
        item = NotificationItem(
            severity=Severity.CRITICAL,
            channel=ChannelRoute.ALERTS,
            embed=embed,
        )
        await discord_tasks.notification_queue.enqueue(item)
        # 짧은 drain 대기 — crash 시 전송 보장
        await asyncio.sleep(2)
