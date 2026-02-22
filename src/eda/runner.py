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
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
    ) -> None:
        self._strategy = strategy
        self._config = config
        self._initial_capital = initial_capital
        self._asset_weights = asset_weights
        self._queue_size = queue_size
        self._target_timeframe = target_timeframe
        self._regime_config = regime_config
        self._feature_store_config = feature_store_config

        # feed/executor 생성
        self._feed: DataFeedPort = HistoricalDataFeed(data, target_timeframe=target_timeframe)
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
        regime_config: RegimeServiceConfig | None = None,
        feature_store_config: FeatureStoreConfig | None = None,
    ) -> EDARunner:
        """백테스트용 Runner 생성.

        HistoricalDataFeed(1m→target_tf) + BacktestExecutor 조합입니다.
        """
        return cls._from_adapters(
            strategy=strategy,
            feed=HistoricalDataFeed(data, target_timeframe=target_timeframe),
            executor=BacktestExecutor(cost_model=config.cost_model),
            target_timeframe=target_timeframe,
            config=config,
            initial_capital=initial_capital,
            asset_weights=asset_weights,
            queue_size=queue_size,
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

        # 모든 enrichment provider + StrategyEngine 생성
        strategy_engine_kwargs = self._build_strategy_engine_kwargs()

        # register(bus) 호출이 필요한 컴포넌트를 별도 참조 (run() 메서드에서 사용)
        regime_service: RegimeService | None = strategy_engine_kwargs.get("regime_service")  # type: ignore[assignment]
        feature_store: FeatureStore | None = strategy_engine_kwargs.get("feature_store")  # type: ignore[assignment]

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
        # 백테스트: 동적 max_order_size (equity 성장에 비례 — VBT parity)
        if isinstance(executor, BacktestExecutor):
            rm.enable_dynamic_max_order_size()
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
                    fills = bt_executor.drain_fills()
                    if fills:
                        # Fill을 PM에 즉시 반영 → _on_bar의 MTM/SL/TS가 최신 포지션 기준
                        await pm.apply_fills(fills)
                    for fill in fills:
                        await bus.publish(fill)  # analytics, rm 등 다른 구독자용

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
        logger.info("EDA Runner starting...")
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

    def _build_strategy_engine_kwargs(self) -> dict[str, object]:
        """StrategyEngine kwargs 구성 — 모든 enrichment provider 생성.

        Returns:
            StrategyEngine 생성자에 전달할 kwargs dict
        """
        kwargs: dict[str, object] = {
            "target_timeframe": self._target_timeframe,
        }

        # 각 provider를 생성하고 None이 아니면 추가
        deriv_provider = self._create_derivatives_provider()
        providers: dict[str, object | None] = {
            "regime_service": self._create_regime_service(deriv_provider),
            "derivatives_provider": deriv_provider,
            "onchain_provider": self._create_onchain_provider(),
            "feature_store": self._create_feature_store(),
            "macro_provider": self._create_macro_provider(),
            "options_provider": self._create_options_provider(),
            "deriv_ext_provider": self._create_deriv_ext_provider(),
        }
        kwargs.update({k: v for k, v in providers.items() if v is not None})

        return kwargs

    def _create_regime_service(
        self,
        derivatives_provider: object | None = None,
    ) -> RegimeService | None:
        """RegimeService 생성 + 전체 데이터 사전 계산.

        regime_config가 None이면 None을 반환합니다.

        Args:
            derivatives_provider: 파생상품 데이터 프로바이더 (있으면 DerivativesDetector 활성화)

        Returns:
            RegimeService 또는 None
        """
        if self._regime_config is None:
            return None

        from src.data.market_data import MarketDataSet, MultiSymbolData
        from src.eda.analytics import tf_to_pandas_freq
        from src.regime.service import RegimeService

        regime_service = RegimeService(
            self._regime_config, derivatives_provider=derivatives_provider
        )

        feed = self._feed
        if not isinstance(feed, HistoricalDataFeed):
            return regime_service

        data = feed.data
        freq = tf_to_pandas_freq(self._target_timeframe)

        # derivatives_provider에서 symbol별 deriv_df 추출
        deriv_map: dict[str, pd.DataFrame] = {}
        if derivatives_provider is not None and hasattr(derivatives_provider, "_precomputed"):
            deriv_map = derivatives_provider._precomputed  # type: ignore[union-attr]

        if isinstance(data, MarketDataSet):
            df_tf = resample_1m_to_tf(data.ohlcv, freq)
            regime_service.precompute(
                data.symbol,
                df_tf["close"],
                deriv_df=deriv_map.get(data.symbol),  # type: ignore[arg-type]
            )
        else:
            assert isinstance(data, MultiSymbolData)
            for sym in data.symbols:
                df_tf = resample_1m_to_tf(data.ohlcv[sym], freq)
                regime_service.precompute(
                    sym,
                    df_tf["close"],
                    deriv_df=deriv_map.get(sym),  # type: ignore[arg-type]
                )

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

    def _create_macro_provider(self) -> object | None:
        """BacktestMacroProvider 생성 (Silver macro 있을 때만).

        GLOBAL scope: 아무 심볼의 resampled index 사용.

        Returns:
            BacktestMacroProvider 또는 None
        """
        from src.data.macro.service import MacroDataService
        from src.data.market_data import MarketDataSet, MultiSymbolData
        from src.eda.analytics import tf_to_pandas_freq
        from src.eda.macro_feed import BacktestMacroProvider

        feed = self._feed
        if not isinstance(feed, HistoricalDataFeed):
            return None

        data = feed.data
        freq = tf_to_pandas_freq(self._target_timeframe)

        # GLOBAL이므로 아무 심볼의 index 사용
        if isinstance(data, MarketDataSet):
            df_tf = resample_1m_to_tf(data.ohlcv, freq)
            ohlcv_index = df_tf.index
        else:
            assert isinstance(data, MultiSymbolData)
            first_sym = data.symbols[0]
            df_tf = resample_1m_to_tf(data.ohlcv[first_sym], freq)
            ohlcv_index = df_tf.index

        service = MacroDataService()
        macro = service.precompute(ohlcv_index)

        if macro.empty or macro.columns.empty or macro.dropna(how="all").empty:
            return None

        logger.info("Macro provider created ({} columns)", len(macro.columns))
        return BacktestMacroProvider(macro)

    def _create_options_provider(self) -> object | None:
        """BacktestOptionsProvider 생성 (Silver options 있을 때만).

        GLOBAL scope: 아무 심볼의 resampled index 사용.

        Returns:
            BacktestOptionsProvider 또는 None
        """
        from src.data.market_data import MarketDataSet, MultiSymbolData
        from src.data.options.service import OptionsDataService
        from src.eda.analytics import tf_to_pandas_freq
        from src.eda.options_feed import BacktestOptionsProvider

        feed = self._feed
        if not isinstance(feed, HistoricalDataFeed):
            return None

        data = feed.data
        freq = tf_to_pandas_freq(self._target_timeframe)

        # GLOBAL이므로 아무 심볼의 index 사용
        if isinstance(data, MarketDataSet):
            df_tf = resample_1m_to_tf(data.ohlcv, freq)
            ohlcv_index = df_tf.index
        else:
            assert isinstance(data, MultiSymbolData)
            first_sym = data.symbols[0]
            df_tf = resample_1m_to_tf(data.ohlcv[first_sym], freq)
            ohlcv_index = df_tf.index

        service = OptionsDataService()
        options = service.precompute(ohlcv_index)

        if options.empty or options.columns.empty or options.dropna(how="all").empty:
            return None

        logger.info("Options provider created ({} columns)", len(options.columns))
        return BacktestOptionsProvider(options)

    def _create_deriv_ext_provider(self) -> object | None:
        """BacktestDerivExtProvider 생성 (Silver deriv_ext 있을 때만).

        PER-ASSET scope: symbol별 독립 precompute.

        Returns:
            BacktestDerivExtProvider 또는 None
        """
        from src.data.deriv_ext.service import DerivExtDataService
        from src.data.market_data import MarketDataSet, MultiSymbolData
        from src.eda.analytics import tf_to_pandas_freq
        from src.eda.deriv_ext_feed import BacktestDerivExtProvider

        feed = self._feed
        if not isinstance(feed, HistoricalDataFeed):
            return None

        data = feed.data
        freq = tf_to_pandas_freq(self._target_timeframe)
        service = DerivExtDataService()
        precomputed: dict[str, pd.DataFrame] = {}

        if isinstance(data, MarketDataSet):
            df_tf = resample_1m_to_tf(data.ohlcv, freq)
            asset = data.symbol.split("/")[0].upper()
            dext = service.precompute(df_tf.index, asset=asset)
            if not dext.empty and not dext.dropna(how="all").empty:
                precomputed[data.symbol] = dext
        else:
            assert isinstance(data, MultiSymbolData)
            for sym in data.symbols:
                df_tf = resample_1m_to_tf(data.ohlcv[sym], freq)
                asset = sym.split("/")[0].upper()
                dext = service.precompute(df_tf.index, asset=asset)
                if not dext.empty and not dext.dropna(how="all").empty:
                    precomputed[sym] = dext

        if not precomputed:
            return None

        logger.info(
            "DerivExt provider created for {} symbols",
            len(precomputed),
        )
        return BacktestDerivExtProvider(precomputed)

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
