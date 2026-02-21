"""OrchestratedRunner — 멀티 전략 오케스트레이터 EDA 백테스트 실행기.

StrategyOrchestrator를 EDA 백테스트 파이프라인에 통합합니다.
EDARunner의 StrategyEngine 자리를 Orchestrator가 대체하여
멀티 전략 시그널을 넷팅 → 단일 포트폴리오로 실행합니다.

Rules Applied:
    - EDARunner.run()과 동일한 컴포넌트 등록 패턴
    - Deferred execution (BacktestExecutor)
    - EventBus flush 필수
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.core.event_bus import EventBus
from src.core.events import AnyEvent, BarEvent, EventType
from src.eda.analytics import AnalyticsEngine
from src.eda.data_feed import HistoricalDataFeed
from src.eda.executors import BacktestExecutor
from src.eda.oms import OMS
from src.eda.portfolio_manager import EDAPortfolioManager
from src.eda.risk_manager import EDARiskManager
from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.lifecycle import LifecycleManager
from src.orchestrator.orchestrator import StrategyOrchestrator
from src.orchestrator.pod import StrategyPod, build_pods
from src.orchestrator.result import OrchestratedResult
from src.orchestrator.risk_aggregator import RiskAggregator
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel

if TYPE_CHECKING:
    from src.data.market_data import MultiSymbolData
    from src.orchestrator.config import OrchestratorConfig

# ── Constants ─────────────────────────────────────────────────────
_MIN_STD_THRESHOLD = 1e-12
_PM_STOP_LOSS_HEADROOM = 1.5
_PM_STOP_LOSS_CAP = 0.30


def _derive_pm_config(orch_config: OrchestratorConfig) -> PortfolioManagerConfig:
    """OrchestratorConfig에서 PM config를 유도합니다.

    Orchestrator가 리스크를 관리하므로 PM은 safety-net 가드레일만 설정합니다.
    PM stop-loss = max_portfolio_drawdown * 1.5 (cap 30%).
    TS/Rebalance: Pod-level 설정을 집계 (any True → True, max multiplier).

    Args:
        orch_config: OrchestratorConfig

    Returns:
        유도된 PortfolioManagerConfig
    """
    pm_stop_loss = min(
        orch_config.max_portfolio_drawdown * _PM_STOP_LOSS_HEADROOM,
        _PM_STOP_LOSS_CAP,
    )

    # Pod-level TS/rebalance 설정 집계
    use_ts = any(p.use_trailing_stop for p in orch_config.pods)
    ts_multiplier = max(p.trailing_stop_atr_multiplier for p in orch_config.pods)
    rebalance_threshold = min(p.rebalance_threshold for p in orch_config.pods)

    return PortfolioManagerConfig(
        max_leverage_cap=orch_config.max_gross_leverage,
        system_stop_loss=pm_stop_loss,
        use_trailing_stop=use_ts,
        trailing_stop_atr_multiplier=ts_multiplier,
        rebalance_threshold=rebalance_threshold,
        cash_sharing=True,
        cost_model=CostModel(
            taker_fee=orch_config.cost_bps / 10000,
            maker_fee=orch_config.cost_bps / 10000,
            slippage=0.0001,
        ),
    )


def _compute_pod_metrics(daily_returns: list[float]) -> dict[str, float]:
    """일별 수익률 리스트에서 간이 메트릭을 계산합니다.

    Args:
        daily_returns: 일별 수익률 리스트

    Returns:
        간이 메트릭 dict (sharpe, mdd, total_return, n_days)
    """
    if not daily_returns:
        return {"sharpe": 0.0, "mdd": 0.0, "total_return": 0.0, "n_days": 0}

    import numpy as np

    returns = np.array(daily_returns)
    total_return = float(np.prod(1 + returns) - 1)
    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1)) if len(returns) > 1 else 1.0
    annualized_factor = 365**0.5
    sharpe = (mean_r / std_r * annualized_factor) if std_r > _MIN_STD_THRESHOLD else 0.0

    # MDD
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    mdd = float(np.min(drawdowns))

    return {
        "sharpe": round(sharpe, 4),
        "mdd": round(mdd, 4),
        "total_return": round(total_return, 4),
        "n_days": len(daily_returns),
    }


class OrchestratedRunner:
    """멀티 전략 오케스트레이터 EDA 백테스트 실행기.

    StrategyOrchestrator를 EDA 파이프라인에 통합하여
    멀티 전략 시그널 넷팅 → 포트폴리오 실행을 수행합니다.

    Args:
        orchestrator_config: OrchestratorConfig
        data: MultiSymbolData (1m)
        target_timeframe: 집계 목표 TF ("1D", "4h" 등)
        pm_config: PM 설정 (None이면 자동 유도)
        initial_capital: 초기 자본 (USD)
        queue_size: EventBus 큐 크기
        pods: 외부 주입 Pod 리스트 (None이면 build_pods 사용)
    """

    def __init__(
        self,
        orchestrator_config: OrchestratorConfig,
        data: MultiSymbolData,
        target_timeframe: str = "1D",
        pm_config: PortfolioManagerConfig | None = None,
        initial_capital: float = 100_000.0,
        queue_size: int = 10000,
        pods: list[StrategyPod] | None = None,
    ) -> None:
        self._orch_config = orchestrator_config
        self._data = data
        self._target_timeframe = target_timeframe
        self._pm_config = pm_config or _derive_pm_config(orchestrator_config)
        self._initial_capital = initial_capital
        self._queue_size = queue_size
        self._pods = pods

        # Multi-TF: Pod configs에서 필요 TF 자동 수집
        self._all_timeframes: set[str] = set(orchestrator_config.all_timeframes)

    @classmethod
    def backtest(
        cls,
        orchestrator_config: OrchestratorConfig,
        data: MultiSymbolData,
        target_timeframe: str = "1D",
        pm_config: PortfolioManagerConfig | None = None,
        initial_capital: float = 100_000.0,
        queue_size: int = 10000,
        pods: list[StrategyPod] | None = None,
    ) -> OrchestratedRunner:
        """백테스트용 OrchestratedRunner 생성.

        Args:
            orchestrator_config: OrchestratorConfig
            data: MultiSymbolData (1m)
            target_timeframe: 집계 목표 TF
            pm_config: PM 설정 (None이면 자동 유도)
            initial_capital: 초기 자본
            queue_size: EventBus 큐 크기
            pods: 외부 주입 Pod 리스트

        Returns:
            OrchestratedRunner 인스턴스
        """
        return cls(
            orchestrator_config=orchestrator_config,
            data=data,
            target_timeframe=target_timeframe,
            pm_config=pm_config,
            initial_capital=initial_capital,
            queue_size=queue_size,
            pods=pods,
        )

    async def run(self) -> OrchestratedResult:
        """EDA 오케스트레이터 백테스트 실행.

        Returns:
            OrchestratedResult — 포트폴리오 + Pod별 메트릭/equity curve
        """
        orch_config = self._orch_config
        pm_config = self._pm_config

        # 1. Pods 생성
        pods = self._pods if self._pods is not None else build_pods(orch_config)

        # 2. Orchestrator 의존성 생성
        allocator = CapitalAllocator(config=orch_config)
        lifecycle_mgr = LifecycleManager(
            graduation=orch_config.graduation,
            retirement=orch_config.retirement,
        )
        risk_aggregator = RiskAggregator(
            config=orch_config,
        )

        # Multi-TF: per-pod routing (target_timeframe=None) vs 단일 TF (기존 경로)
        multi_tf = len(self._all_timeframes) > 1
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=pods,
            allocator=allocator,
            lifecycle_manager=lifecycle_mgr,
            risk_aggregator=risk_aggregator,
            target_timeframe=None if multi_tf else self._target_timeframe,
        )

        # 2.5. Initial capital → Pod base equity 초기화
        orchestrator.set_initial_capital(self._initial_capital)

        # 3. EDA 컴포넌트 생성
        bus = EventBus(queue_size=self._queue_size)
        feed = HistoricalDataFeed(
            self._data,
            target_timeframe=self._target_timeframe,
            target_timeframes=self._all_timeframes if multi_tf else None,
        )
        executor = BacktestExecutor(cost_model=pm_config.cost_model)

        # asset_weights: Pod이 1/N 정규화 완료 → PM은 uniform 1.0
        all_symbols = list(orch_config.all_symbols)
        asset_weights = dict.fromkeys(all_symbols, 1.0)

        # PM: 가장 짧은 TF 사용 (intrabar SL/TS 리셋 주기)
        from src.eda.candle_aggregator import timeframe_to_seconds

        shortest_tf = min(
            self._all_timeframes,
            key=lambda tf: timeframe_to_seconds(tf),
        )
        pm = EDAPortfolioManager(
            config=pm_config,
            initial_capital=self._initial_capital,
            asset_weights=asset_weights,
            target_timeframe=shortest_tf,
        )
        rm = EDARiskManager(
            config=pm_config,
            portfolio_manager=pm,
            max_order_size_usd=self._initial_capital * pm_config.max_leverage_cap,
            enable_circuit_breaker=False,
        )
        oms = OMS(executor=executor, portfolio_manager=pm)
        analytics = AnalyticsEngine(initial_capital=self._initial_capital)

        # 4. 컴포넌트 등록 (순서 중요!)
        # 4a. Executor bar handler (BAR → deferred fill)
        all_timeframes = {pod.timeframe for pod in pods}

        async def executor_bar_handler(event: AnyEvent) -> None:
            assert isinstance(event, BarEvent)
            executor.on_bar(event)
            if event.timeframe in all_timeframes:
                executor.fill_pending(event)
                for fill in executor.drain_fills():
                    await bus.publish(fill)

        bus.subscribe(EventType.BAR, executor_bar_handler)

        # 4b. Orchestrator (BAR[TF only] → pod routing → SIGNAL)
        await orchestrator.register(bus)

        # 4c. PM → RM → OMS → Analytics
        await pm.register(bus)
        await rm.register(bus)
        await oms.register(bus)
        await analytics.register(bus)

        # 5. 실행
        tf_label = (
            ", ".join(sorted(self._all_timeframes))
            if multi_tf
            else self._target_timeframe
        )
        logger.info(
            "OrchestratedRunner starting: {} pods, {} symbols, TF={}{}",
            len(pods),
            len(all_symbols),
            tf_label,
            " (multi-TF)" if multi_tf else "",
        )
        bus_task = asyncio.create_task(bus.start())
        await feed.start(bus)

        # 데이터 종료 후 미처리 시그널 flush
        await orchestrator.flush_pending_signals()
        await pm.flush_pending_signals()
        await bus.flush()

        # 마지막 일 daily return flush
        orchestrator.flush_daily_returns()

        # Deferred execution: 마지막 bar 이후 미체결 pending orders 로깅
        if executor.pending_count > 0:
            logger.info(
                "Discarded {} pending orders at end of backtest (no next bar)",
                executor.pending_count,
            )

        await bus.stop()
        await bus_task

        # 6. 결과 생성
        portfolio_metrics = analytics.compute_metrics(
            timeframe=self._target_timeframe,
            cost_model=pm_config.cost_model,
        )
        portfolio_equity = analytics.get_equity_series()

        # Pod equity curves + metrics
        pod_equity_curves = self._build_pod_equity_curves(pods)
        pod_metrics = {pod.pod_id: _compute_pod_metrics(pod.daily_returns) for pod in pods}

        # Allocation history → DataFrame
        allocation_df = (
            pd.DataFrame(orchestrator.allocation_history)
            if orchestrator.allocation_history
            else None
        )

        # Risk contributions → DataFrame
        risk_df = (
            pd.DataFrame(orchestrator.risk_contributions_history)
            if orchestrator.risk_contributions_history
            else None
        )

        logger.info(
            "OrchestratedRunner finished: {} bars, {} fills, {} trades",
            feed.bars_emitted,
            analytics.total_fills,
            portfolio_metrics.total_trades,
        )

        return OrchestratedResult(
            portfolio_metrics=portfolio_metrics,
            portfolio_equity_curve=portfolio_equity,
            pod_metrics=pod_metrics,
            pod_equity_curves=pod_equity_curves,
            allocation_history=allocation_df,
            lifecycle_events=orchestrator.lifecycle_events,
            risk_contributions=risk_df,
        )

    @staticmethod
    def _build_pod_equity_curves(pods: list[StrategyPod]) -> dict[str, pd.Series]:
        """Pod별 daily_returns → cumulative equity curve 재구성.

        Args:
            pods: StrategyPod 리스트

        Returns:
            pod_id → equity curve Series
        """
        import numpy as np

        result: dict[str, pd.Series] = {}
        for pod in pods:
            returns = pod.daily_returns
            if not returns:
                result[pod.pod_id] = pd.Series(dtype=float)
                continue
            cumulative = np.cumprod(1 + np.array(returns))
            result[pod.pod_id] = pd.Series(cumulative, dtype=float)
        return result

    @staticmethod
    def derive_pm_config(orch_config: OrchestratorConfig) -> PortfolioManagerConfig:
        """OrchestratorConfig에서 PM config를 유도합니다 (외부 접근용).

        Args:
            orch_config: OrchestratorConfig

        Returns:
            유도된 PortfolioManagerConfig
        """
        return _derive_pm_config(orch_config)
