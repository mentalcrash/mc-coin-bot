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
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType
from src.eda.analytics import AnalyticsEngine
from src.eda.executors import BacktestExecutor, LiveExecutor, ShadowExecutor
from src.eda.live_data_feed import LiveDataFeed
from src.eda.oms import OMS
from src.eda.portfolio_manager import EDAPortfolioManager
from src.eda.risk_manager import EDARiskManager
from src.eda.strategy_engine import StrategyEngine

if TYPE_CHECKING:
    from src.eda.persistence.database import Database
    from src.eda.persistence.state_manager import StateManager
    from src.eda.ports import ExecutorPort
    from src.eda.reconciler import PositionReconciler
    from src.exchange.binance_client import BinanceClient
    from src.exchange.binance_futures_client import BinanceFuturesClient
    from src.monitoring.metrics import MetricsExporter
    from src.notification.bot import DiscordBotService
    from src.notification.config import DiscordBotConfig
    from src.notification.queue import NotificationQueue
    from src.notification.report_scheduler import ReportScheduler
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
        self._client: BinanceClient | None = None
        self._futures_client: BinanceFuturesClient | None = None
        self._symbols: list[str] = []

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
        )
        runner._client = client
        runner._symbols = symbols
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
        )
        runner._client = client
        runner._symbols = symbols
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
        )
        runner._client = client
        runner._futures_client = futures_client
        runner._symbols = symbols
        return runner

    async def run(self) -> None:
        """메인 루프. shutdown_event까지 실행."""
        from src.eda.persistence.database import Database
        from src.eda.persistence.trade_persistence import TradePersistence

        # 0. LIVE 모드 Pre-flight checks → 거래소 잔고로 capital override
        capital = self._initial_capital
        if self._mode == LiveMode.LIVE and self._futures_client is not None:
            capital = await self._preflight_checks()

        # 1. DB 초기화 (db_path가 있을 때만)
        db: Database | None = None
        if self._db_path:
            db = Database(self._db_path)
            await db.connect()

        try:
            # 2. 컴포넌트 생성
            bus = EventBus(queue_size=self._queue_size)

            # RegimeService (선택적)
            regime_service = self._create_regime_service()

            strategy_engine = StrategyEngine(
                self._strategy,
                target_timeframe=self._target_timeframe,
                regime_service=regime_service,
            )
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
            state_mgr = await self._restore_state(db, pm, rm)

            # 4. 모든 컴포넌트 등록 (순서 중요)
            # RegimeService는 StrategyEngine 앞에 등록 (BAR 처리 시 regime 먼저 업데이트)
            if regime_service is not None:
                await regime_service.register(bus)
            await strategy_engine.register(bus)
            await pm.register(bus)
            await rm.register(bus)
            await oms.register(bus)
            await analytics.register(bus)

            # 4.5. REST API warmup — WebSocket 시작 전 버퍼 사전 채움
            await self._warmup_strategy(strategy_engine, regime_service=regime_service)

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

            # 주기적 메트릭 갱신 task (uptime + EventBus + exchange health)
            uptime_task: asyncio.Task[None] | None = None
            if metrics_exporter is not None:
                uptime_task = asyncio.create_task(
                    self._periodic_metrics_update(
                        metrics_exporter, bus, self._futures_client
                    )
                )

            # Live 모드: PositionReconciler 초기 + 주기적 검증
            reconciler_task = await self._setup_reconciler(pm, rm)

            # 구조화된 startup summary
            self._log_startup_summary(capital)

            logger.info("LiveRunner started (mode={}, db={})", self._mode.value, self._db_path)

            # 8. Shutdown 대기
            await self._shutdown_event.wait()

            # 9. Graceful shutdown
            logger.warning("Initiating graceful shutdown...")

            # 주기적 task 취소
            for task in (save_task, uptime_task, reconciler_task):
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

    def _create_regime_service(self) -> RegimeService | None:
        """RegimeService 생성 (regime_config가 None이면 None)."""
        if self._regime_config is None:
            return None

        from src.regime.service import RegimeService

        return RegimeService(self._regime_config)

    async def _warmup_strategy(
        self,
        strategy_engine: StrategyEngine,
        *,
        regime_service: RegimeService | None = None,
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
                else:
                    logger.warning("No warmup data fetched for {}", symbol)
            except Exception:
                logger.exception("Warmup failed for {}, continuing without warmup", symbol)

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

        # CCXT TF 변환: Binance는 소문자 사용 (1D → 1d)
        ccxt_tf = timeframe.lower() if timeframe.endswith("D") else timeframe

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

    async def _setup_reconciler(
        self,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
    ) -> asyncio.Task[None] | None:
        """Live 모드: PositionReconciler 초기 + 주기적 검증 task 생성."""
        if self._mode != LiveMode.LIVE or self._futures_client is None:
            return None

        from src.eda.reconciler import PositionReconciler

        reconciler = PositionReconciler()
        await reconciler.initial_check(pm, self._futures_client, self._symbols)
        return asyncio.create_task(
            self._periodic_reconciliation(reconciler, pm, rm, self._futures_client, self._symbols)
        )

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

        # 3. 미체결 주문 감지
        for symbol in self._symbols:
            futures_symbol = self._futures_client.to_futures_symbol(symbol)
            try:
                open_orders = await self._futures_client.fetch_open_orders(futures_symbol)
                if open_orders:
                    logger.critical(
                        "Pre-flight: {} stale open orders for {} — cancel manually before trading!",
                        len(open_orders),
                        symbol,
                    )
            except Exception:
                logger.warning("Pre-flight: Failed to check open orders for {}", symbol)

        logger.info("Pre-flight checks PASSED — using exchange balance ${:.2f}", total_balance)
        return total_balance

    def _log_startup_summary(self, capital: float) -> None:
        """구조화된 startup summary 로그."""
        summary = (
            f"=== LiveRunner Startup Summary ===\n"
            f"  Mode:       {self._mode.value}\n"
            f"  Strategy:   {self._strategy.name}\n"
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

        bot_info, trading_mode_enum을 설정합니다.

        Returns:
            MetricsExporter 또는 None (비활성 시)
        """
        if self._metrics_port <= 0:
            return None

        from src.monitoring.metrics import (
            MetricsExporter,
            PrometheusApiCallback,
            PrometheusWsCallback,
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
        if self._futures_client is not None:
            self._futures_client.set_metrics_callback(PrometheusApiCallback())

        # LiveDataFeed에 WS 상태 콜백 주입
        self._feed.set_ws_callback(PrometheusWsCallback())

        return exporter

    @staticmethod
    async def _periodic_metrics_update(
        exporter: MetricsExporter,
        bus: EventBus,
        futures_client: BinanceFuturesClient | None,
        interval: float = 30.0,
    ) -> None:
        """주기적으로 uptime + EventBus + exchange health 메트릭 갱신."""
        while True:
            await asyncio.sleep(interval)
            exporter.update_uptime()
            exporter.update_eventbus_metrics(bus)
            if futures_client is not None:
                exporter.update_exchange_health(futures_client.consecutive_failures)

    @staticmethod
    async def _periodic_reconciliation(
        reconciler: PositionReconciler,
        pm: EDAPortfolioManager,
        rm: EDARiskManager,
        futures_client: BinanceFuturesClient,
        symbols: list[str],
        interval: float = _RECONCILER_INTERVAL,
    ) -> None:
        """주기적 포지션 + 잔고 교차 검증."""
        while True:
            await asyncio.sleep(interval)
            await reconciler.periodic_check(pm, futures_client, symbols)
            # 잔고 검증 + RM equity sync
            exchange_equity = await reconciler.check_balance(pm, futures_client)
            if exchange_equity is not None:
                await rm.sync_exchange_equity(exchange_equity)

    @property
    def feed(self) -> LiveDataFeed:
        """LiveDataFeed 인스턴스."""
        return self._feed

    @property
    def mode(self) -> LiveMode:
        """실행 모드."""
        return self._mode
