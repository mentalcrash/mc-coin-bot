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
from src.eda.data_feed import HistoricalDataFeed, resample_1m_to_tf
from src.eda.executors import BacktestExecutor, ShadowExecutor
from src.eda.oms import OMS
from src.eda.portfolio_manager import EDAPortfolioManager
from src.eda.risk_manager import EDARiskManager
from src.eda.strategy_engine import StrategyEngine

if TYPE_CHECKING:
    import pandas as pd

    from src.data.market_data import MarketDataSet, MultiSymbolData
    from src.eda.ports import DataFeedPort, ExecutorPort
    from src.market.feature_store import FeatureStore, FeatureStoreConfig
    from src.models.backtest import PerformanceMetrics
    from src.portfolio.config import PortfolioManagerConfig
    from src.regime.service import RegimeService, RegimeServiceConfig
    from src.strategy.base import BaseStrategy


class EDARunner:
    """EDA 백테스트 실행기.

    모든 컴포넌트를 조립하고 asyncio 이벤트 루프에서 실행합니다.

    Args:
        strategy: 전략 인스턴스
        data: 단일 또는 멀티 심볼 데이터 (1m)
        target_timeframe: 집계 목표 TF ("1D", "4h", "1h" 등)
        config: 포트폴리오 설정
        initial_capital: 초기 자본 (USD)
        asset_weights: 에셋별 가중치 (None이면 균등분배)
        queue_size: EventBus 큐 크기
        fast_mode: True면 pre-aggregation + incremental 전략 (intrabar SL/TS 없음)
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        target_timeframe: str,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        queue_size: int = 10000,
        *,
        fast_mode: bool = False,
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._initial_capital = initial_capital
        self._asset_weights = asset_weights
        self._queue_size = queue_size
        self._target_timeframe = target_timeframe
        self._fast_mode = fast_mode
        self._regime_config = regime_config
        self._feature_store_config = feature_store_config

        # feed/executor 생성
        self._feed: DataFeedPort = HistoricalDataFeed(
            data, target_timeframe=target_timeframe, fast_mode=fast_mode
        )
        self._executor: ExecutorPort = BacktestExecutor(cost_model=config.cost_model)

        # Components (run() 시 초기화)
        self._bus: EventBus | None = None
        self._analytics: AnalyticsEngine | None = None
        self._pm: EDAPortfolioManager | None = None

    @classmethod
    def _from_adapters(
        cls,
        strategy: BaseStrategy,
        feed: DataFeedPort,
        executor: ExecutorPort,
        target_timeframe: str,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        queue_size: int = 10000,
        *,
        fast_mode: bool = False,
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
    ) -> EDARunner:
        """어댑터를 직접 주입하여 Runner를 생성합니다 (내부용)."""
        instance = object.__new__(cls)
        instance._strategy = strategy
        instance._feed = feed
        instance._executor = executor
        instance._config = config
        instance._initial_capital = initial_capital
        instance._asset_weights = asset_weights
        instance._queue_size = queue_size
        instance._target_timeframe = target_timeframe
        instance._fast_mode = fast_mode
        instance._regime_config = regime_config
        instance._feature_store_config = feature_store_config
        instance._bus = None
        instance._analytics = None
        instance._pm = None
        return instance

    @classmethod
    def backtest(
        cls,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        target_timeframe: str,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        queue_size: int = 10000,
        *,
        fast_mode: bool = False,
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
    ) -> EDARunner:
        """백테스트용 Runner 생성.

        HistoricalDataFeed(1m→target_tf) + BacktestExecutor 조합입니다.
        fast_mode=True면 pre-aggregation + incremental 전략으로 고속 실행.
        """
        return cls._from_adapters(
            strategy=strategy,
            feed=HistoricalDataFeed(data, target_timeframe=target_timeframe, fast_mode=fast_mode),
            executor=BacktestExecutor(cost_model=config.cost_model),
            target_timeframe=target_timeframe,
            config=config,
            initial_capital=initial_capital,
            asset_weights=asset_weights,
            queue_size=queue_size,
            fast_mode=fast_mode,
            regime_config=regime_config,
            feature_store_config=feature_store_config,
        )

    @classmethod
    def shadow(
        cls,
        strategy: BaseStrategy,
        data: MarketDataSet | MultiSymbolData,
        target_timeframe: str,
        config: PortfolioManagerConfig,
        initial_capital: float = 10000.0,
        asset_weights: dict[str, float] | None = None,
        *,
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
    ) -> EDARunner:
        """Shadow 모드 Runner 생성.

        HistoricalDataFeed(1m→target_tf) + ShadowExecutor (로깅만, 체결 없음) 조합입니다.
        """
        return cls._from_adapters(
            strategy=strategy,
            feed=HistoricalDataFeed(data, target_timeframe=target_timeframe),
            executor=ShadowExecutor(),
            target_timeframe=target_timeframe,
            config=config,
            initial_capital=initial_capital,
            asset_weights=asset_weights,
            regime_config=regime_config,
            feature_store_config=feature_store_config,
        )

    async def run(self) -> PerformanceMetrics:
        """EDA 백테스트 실행.

        Returns:
            PerformanceMetrics 결과
        """
        # 1. 컴포넌트 생성
        bus = EventBus(queue_size=self._queue_size)
        self._bus = bus

        feed = self._feed
        executor = self._executor

        # RegimeService 생성 + 사전계산 (regime_config가 있을 때만)
        regime_service = self._create_regime_service()

        # Derivatives provider (선택적)
        derivatives_provider = self._create_derivatives_provider()

        # On-chain provider (선택적 — Silver 데이터 auto-detect)
        onchain_provider = self._create_onchain_provider()

        # FeatureStore (선택적)
        feature_store = self._create_feature_store()

        # fast_mode: signal pre-computation (전체 데이터로 한번에 시그널 계산)
        strategy_engine_kwargs: dict[str, object] = {
            "target_timeframe": self._target_timeframe,
        }
        if self._fast_mode:
            precomputed = self._precompute_signals()
            if precomputed:
                strategy_engine_kwargs["precomputed_signals"] = precomputed
        if regime_service is not None:
            strategy_engine_kwargs["regime_service"] = regime_service
        if derivatives_provider is not None:
            strategy_engine_kwargs["derivatives_provider"] = derivatives_provider
        if onchain_provider is not None:
            strategy_engine_kwargs["onchain_provider"] = onchain_provider
        if feature_store is not None:
            strategy_engine_kwargs["feature_store"] = feature_store

        strategy_engine = StrategyEngine(self._strategy, **strategy_engine_kwargs)  # type: ignore[arg-type]
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
            enable_circuit_breaker=False,
        )
        oms = OMS(executor=executor, portfolio_manager=pm)
        analytics = AnalyticsEngine(initial_capital=self._initial_capital)

        self._analytics = analytics
        self._pm = pm

        # Executor에 bar 가격 업데이트 + deferred fill 발행 핸들러 등록
        if isinstance(executor, BacktestExecutor):
            bt_executor = executor
            target_tf = self._target_timeframe

            async def executor_bar_handler(event: AnyEvent) -> None:
                assert isinstance(event, BarEvent)
                bt_executor.on_bar(event)
                # TF bar에서만 deferred fill 처리 (VBT shift(-1) 동일)
                if event.timeframe == target_tf:
                    bt_executor.fill_pending(event)
                    for fill in bt_executor.drain_fills():
                        await bus.publish(fill)

            bus.subscribe(EventType.BAR, executor_bar_handler)

        # 2. 모든 컴포넌트 등록 (순서 중요)
        # RegimeService → FeatureStore → StrategyEngine 순서
        if regime_service is not None:
            await regime_service.register(bus)
        if feature_store is not None:
            await feature_store.register(bus)
        await strategy_engine.register(bus)
        await pm.register(bus)
        await rm.register(bus)
        await oms.register(bus)
        await analytics.register(bus)

        # 3. 실행
        mode_label = " [fast]" if self._fast_mode else ""
        logger.info("EDA Runner starting...{}", mode_label)
        bus_task = asyncio.create_task(bus.start())

        await feed.start(bus)

        # 마지막 batch flush (데이터 종료 후 미처리 signal 처리)
        await pm.flush_pending_signals()
        await bus.flush()

        # Deferred execution: 마지막 bar 이후 미체결 pending orders 로깅
        if isinstance(executor, BacktestExecutor) and executor.pending_count > 0:
            logger.info(
                "Discarded {} pending orders at end of backtest (no next bar)",
                executor.pending_count,
            )

        await bus.stop()
        await bus_task

        # 4. 결과 생성
        timeframe = self._target_timeframe
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

    def _create_regime_service(self) -> RegimeService | None:
        """RegimeService 생성 + 전체 데이터 사전 계산.

        regime_config가 None이면 None을 반환합니다.

        Returns:
            RegimeService 또는 None
        """
        if self._regime_config is None:
            return None

        from src.data.market_data import MarketDataSet, MultiSymbolData
        from src.eda.analytics import tf_to_pandas_freq
        from src.regime.service import RegimeService

        regime_service = RegimeService(self._regime_config)

        feed = self._feed
        if not isinstance(feed, HistoricalDataFeed):
            return regime_service

        data = feed.data
        freq = tf_to_pandas_freq(self._target_timeframe)

        if isinstance(data, MarketDataSet):
            df_tf = resample_1m_to_tf(data.ohlcv, freq)
            regime_service.precompute(data.symbol, df_tf["close"])  # type: ignore[arg-type]
        else:
            assert isinstance(data, MultiSymbolData)
            for sym in data.symbols:
                df_tf = resample_1m_to_tf(data.ohlcv[sym], freq)
                regime_service.precompute(sym, df_tf["close"])  # type: ignore[arg-type]

        return regime_service

    def _create_derivatives_provider(self) -> object | None:
        """BacktestDerivativesProvider 생성 (Silver _deriv 있을 때만).

        Returns:
            BacktestDerivativesProvider 또는 None
        """
        from src.data.derivatives_service import DerivativesDataService
        from src.data.market_data import MarketDataSet, MultiSymbolData
        from src.eda.analytics import tf_to_pandas_freq
        from src.eda.derivatives_feed import BacktestDerivativesProvider

        feed = self._feed
        if not isinstance(feed, HistoricalDataFeed):
            return None

        data = feed.data
        freq = tf_to_pandas_freq(self._target_timeframe)
        deriv_service = DerivativesDataService()
        precomputed: dict[str, pd.DataFrame] = {}

        if isinstance(data, MarketDataSet):
            df_tf = resample_1m_to_tf(data.ohlcv, freq)
            deriv = deriv_service.precompute(
                data.symbol,
                df_tf.index,
                data.start,
                data.end,  # type: ignore[arg-type]
            )
            if not deriv.empty and not deriv.dropna(how="all").empty:
                precomputed[data.symbol] = deriv
        else:
            assert isinstance(data, MultiSymbolData)
            for sym in data.symbols:
                df_tf = resample_1m_to_tf(data.ohlcv[sym], freq)
                deriv = deriv_service.precompute(
                    sym,
                    df_tf.index,
                    data.start,
                    data.end,  # type: ignore[arg-type]
                )
                if not deriv.empty and not deriv.dropna(how="all").empty:
                    precomputed[sym] = deriv

        if not precomputed:
            return None

        logger.info(
            "Derivatives provider created for {} symbols",
            len(precomputed),
        )
        return BacktestDerivativesProvider(precomputed)

    def _create_onchain_provider(self) -> object | None:
        """BacktestOnchainProvider 생성 (Silver on-chain 있을 때만).

        Silver 데이터가 없으면 자동 skip합니다 (config 불필요).

        Returns:
            BacktestOnchainProvider 또는 None
        """
        from src.data.market_data import MarketDataSet, MultiSymbolData
        from src.data.onchain.service import OnchainDataService
        from src.eda.analytics import tf_to_pandas_freq
        from src.eda.onchain_feed import BacktestOnchainProvider

        feed = self._feed
        if not isinstance(feed, HistoricalDataFeed):
            return None

        data = feed.data
        freq = tf_to_pandas_freq(self._target_timeframe)
        service = OnchainDataService()
        precomputed: dict[str, pd.DataFrame] = {}

        if isinstance(data, MarketDataSet):
            df_tf = resample_1m_to_tf(data.ohlcv, freq)
            onchain = service.precompute(symbol=data.symbol, ohlcv_index=df_tf.index)
            if not onchain.empty and not onchain.dropna(how="all").empty:
                precomputed[data.symbol] = onchain
        else:
            assert isinstance(data, MultiSymbolData)
            for sym in data.symbols:
                df_tf = resample_1m_to_tf(data.ohlcv[sym], freq)
                onchain = service.precompute(symbol=sym, ohlcv_index=df_tf.index)
                if not onchain.empty and not onchain.dropna(how="all").empty:
                    precomputed[sym] = onchain

        if not precomputed:
            return None

        logger.info(
            "On-chain provider created for {} symbols",
            len(precomputed),
        )
        return BacktestOnchainProvider(precomputed)

    def _create_feature_store(self) -> FeatureStore | None:
        """FeatureStore 생성 + 전체 데이터 사전 계산.

        feature_store_config가 None이면 None을 반환합니다.

        Returns:
            FeatureStore 또는 None
        """
        if self._feature_store_config is None:
            return None

        from src.data.market_data import MarketDataSet, MultiSymbolData
        from src.eda.analytics import tf_to_pandas_freq
        from src.market.feature_store import FeatureStore

        store = FeatureStore(self._feature_store_config)

        feed = self._feed
        if not isinstance(feed, HistoricalDataFeed):
            return store

        data = feed.data
        freq = tf_to_pandas_freq(self._target_timeframe)

        if isinstance(data, MarketDataSet):
            df_tf = resample_1m_to_tf(data.ohlcv, freq)
            store.precompute(data.symbol, df_tf)
        else:
            assert isinstance(data, MultiSymbolData)
            for sym in data.symbols:
                df_tf = resample_1m_to_tf(data.ohlcv[sym], freq)
                store.precompute(sym, df_tf)

        return store

    def _precompute_signals(self) -> dict[str, object] | None:
        """fast_mode: 전체 TF 데이터로 시그널을 사전 계산.

        forward_return 등 미래 참조 feature를 사용하는 전략(CTREND)에서
        bar-by-bar 증분 실행 시 발생하는 edge effect를 방지합니다.
        VBT와 동일하게 전체 데이터셋에서 한번에 시그널을 계산합니다.

        Returns:
            {symbol: StrategySignals} 또는 실패 시 None
        """
        from src.data.market_data import MarketDataSet, MultiSymbolData
        from src.eda.analytics import tf_to_pandas_freq

        feed = self._feed
        if not isinstance(feed, HistoricalDataFeed):
            return None

        data = feed.data
        freq = tf_to_pandas_freq(self._target_timeframe)
        result: dict[str, object] = {}

        if isinstance(data, MarketDataSet):
            df_tf = resample_1m_to_tf(data.ohlcv, freq)
            _, signals = self._strategy.run(df_tf)
            result[data.symbol] = signals
            logger.info("Pre-computed signals for {} ({} bars)", data.symbol, len(df_tf))
        else:
            assert isinstance(data, MultiSymbolData)
            for sym in data.symbols:
                df_tf = resample_1m_to_tf(data.ohlcv[sym], freq)
                _, signals = self._strategy.run(df_tf)
                result[sym] = signals
                logger.info("Pre-computed signals for {} ({} bars)", sym, len(df_tf))

        return result if result else None

    @property
    def analytics(self) -> AnalyticsEngine | None:
        """Analytics 엔진 참조 (run() 후 접근 가능)."""
        return self._analytics

    @property
    def portfolio_manager(self) -> EDAPortfolioManager | None:
        """PM 참조 (run() 후 접근 가능)."""
        return self._pm

    @property
    def config(self) -> PortfolioManagerConfig:
        """포트폴리오 설정."""
        return self._config

    @property
    def target_timeframe(self) -> str:
        """타겟 타임프레임."""
        return self._target_timeframe
