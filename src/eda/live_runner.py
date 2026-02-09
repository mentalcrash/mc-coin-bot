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
    from src.eda.persistence.state_manager import StateManager
    from src.eda.ports import ExecutorPort
    from src.exchange.binance_client import BinanceClient
    from src.portfolio.config import PortfolioManagerConfig
    from src.strategy.base import BaseStrategy

# 기본 상태 저장 주기 (초)
_DEFAULT_SAVE_INTERVAL = 300.0


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
        )

    async def run(self) -> None:
        """메인 루프. shutdown_event까지 실행."""
        from src.eda.persistence.database import Database
        from src.eda.persistence.state_manager import StateManager
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
            state_mgr: StateManager | None = None
            if db:
                state_mgr = StateManager(db)
                pm_state = await state_mgr.load_pm_state()
                if pm_state:
                    pm.restore_state(pm_state)
                    logger.info("PM state restored")
                rm_state = await state_mgr.load_rm_state()
                if rm_state:
                    rm.restore_state(rm_state)
                    logger.info("RM state restored")

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

            # 6. Signal handler
            self._setup_signal_handlers()

            # 7. 실행
            bus_task = asyncio.create_task(bus.start())
            feed_task = asyncio.create_task(self._feed.start(bus))

            # 주기적 상태 저장 task
            save_task: asyncio.Task[None] | None = None
            if state_mgr:
                save_task = asyncio.create_task(self._periodic_state_save(state_mgr, pm, rm))

            logger.info("LiveRunner started (mode={}, db={})", self._mode.value, self._db_path)

            # 8. Shutdown 대기
            await self._shutdown_event.wait()

            # 9. Graceful shutdown
            logger.warning("Initiating graceful shutdown...")

            # 주기적 저장 task 취소
            if save_task:
                save_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await save_task

            await self._feed.stop()
            feed_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await feed_task

            await pm.flush_pending_signals()
            await bus.flush()
            await bus.stop()
            await bus_task

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

    @property
    def feed(self) -> LiveDataFeed:
        """LiveDataFeed 인스턴스."""
        return self._feed

    @property
    def mode(self) -> LiveMode:
        """실행 모드."""
        return self._mode
