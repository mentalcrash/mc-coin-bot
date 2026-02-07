"""EDA Runner — 통합 오케스트레이터.

모든 EDA 컴포넌트를 조립하고 asyncio로 실행합니다.
DataFeed → EventBus → StrategyEngine → PM → RM → OMS → Executor → Analytics

Rules Applied:
    - Component Assembly: 모든 컴포넌트를 올바른 순서로 등록
    - Backtest-Live Parity: 동일 인터페이스로 백테스트/라이브 전환
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType
from src.eda.analytics import AnalyticsEngine
from src.eda.data_feed import AggregatingDataFeed, HistoricalDataFeed
from src.eda.executors import BacktestExecutor
from src.eda.oms import OMS
from src.eda.portfolio_manager import EDAPortfolioManager
from src.eda.risk_manager import EDARiskManager
from src.eda.strategy_engine import StrategyEngine

if TYPE_CHECKING:
    from src.data.market_data import MarketDataSet, MultiSymbolData
    from src.models.backtest import PerformanceMetrics
    from src.portfolio.config import PortfolioManagerConfig
    from src.strategy.base import BaseStrategy


class EDARunner:
    """EDA 백테스트 실행기.

    모든 컴포넌트를 조립하고 asyncio 이벤트 루프에서 실행합니다.

    Args:
        strategy: 전략 인스턴스
        data: 단일 또는 멀티 심볼 데이터
        config: 포트폴리오 설정
        initial_capital: 초기 자본 (USD)
        asset_weights: 에셋별 가중치 (None이면 균등분배)
        queue_size: EventBus 큐 크기
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        queue_size: int = 10000,
        target_timeframe: str | None = None,
    ) -> None:
        self._strategy = strategy
        self._data = data
        self._config = config
        self._initial_capital = initial_capital
        self._asset_weights = asset_weights
        self._queue_size = queue_size
        self._target_timeframe = target_timeframe

        # Components (run() 시 초기화)
        self._bus: EventBus | None = None
        self._analytics: AnalyticsEngine | None = None
        self._pm: EDAPortfolioManager | None = None

    async def run(self) -> PerformanceMetrics:
        """EDA 백테스트 실행.

        Returns:
            PerformanceMetrics 결과
        """
        # 1. 컴포넌트 생성
        bus = EventBus(queue_size=self._queue_size)
        self._bus = bus

        # 데이터 피드 선택: target_timeframe 설정 시 1m aggregation 모드
        if self._target_timeframe is not None:
            feed: HistoricalDataFeed | AggregatingDataFeed = AggregatingDataFeed(
                self._data, target_timeframe=self._target_timeframe
            )
        else:
            feed = HistoricalDataFeed(self._data)

        strategy_engine = StrategyEngine(
            self._strategy, target_timeframe=self._target_timeframe
        )
        pm = EDAPortfolioManager(
            config=self._config,
            initial_capital=self._initial_capital,
            asset_weights=self._asset_weights,
            target_timeframe=self._target_timeframe,
        )
        rm = EDARiskManager(config=self._config, portfolio_manager=pm)
        executor = BacktestExecutor(cost_model=self._config.cost_model)
        oms = OMS(executor=executor, portfolio_manager=pm)
        analytics = AnalyticsEngine(initial_capital=self._initial_capital)

        self._analytics = analytics
        self._pm = pm

        # Executor에 bar 가격 업데이트를 위한 핸들러 등록
        async def executor_bar_handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            executor.on_bar(event)

        bus.subscribe(EventType.BAR, executor_bar_handler)

        # 2. 모든 컴포넌트 등록 (순서 중요)
        await strategy_engine.register(bus)
        await pm.register(bus)
        await rm.register(bus)
        await oms.register(bus)
        await analytics.register(bus)

        # 3. 실행
        logger.info("EDA Runner starting...")
        bus_task = asyncio.create_task(bus.start())

        await feed.start(bus)
        await bus.stop()
        await bus_task

        # 4. 결과 생성
        timeframe = self._target_timeframe or self._data.timeframe
        metrics = analytics.compute_metrics(
            timeframe=timeframe,
            cost_model=self._config.cost_model,
        )
        logger.info(
            "EDA Runner finished: {} bars, {} fills, {} trades",
            feed.bars_emitted,
            analytics.total_fills,
            metrics.total_trades,
        )

        return metrics

    @property
    def analytics(self) -> AnalyticsEngine | None:
        """Analytics 엔진 참조 (run() 후 접근 가능)."""
        return self._analytics

    @property
    def portfolio_manager(self) -> EDAPortfolioManager | None:
        """PM 참조 (run() 후 접근 가능)."""
        return self._pm
