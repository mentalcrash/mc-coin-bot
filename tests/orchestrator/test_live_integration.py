"""LiveRunner Orchestrated 모드 통합 테스트.

orchestrated_paper/orchestrated_live classmethod + run() 내부 분기,
warmup, startup summary 등을 검증합니다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from src.eda.live_runner import LiveMode, LiveRunner
from src.orchestrator.allocator import CapitalAllocator
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod
from src.orchestrator.orchestrator import StrategyOrchestrator
from src.orchestrator.pod import StrategyPod
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────────


class SimpleTestStrategy(BaseStrategy):
    """테스트용 간단 전략."""

    @property
    def name(self) -> str:
        return "test_simple"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        direction = (df["close"] > df["open"]).astype(int) * 2 - 1
        strength = pd.Series(0.5, index=df.index)
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


# ── Helpers ───────────────────────────────────────────────────────


def _make_pod_config(
    pod_id: str,
    symbols: tuple[str, ...],
    initial_fraction: float = 0.5,
) -> PodConfig:
    return PodConfig(
        pod_id=pod_id,
        strategy_name="tsmom",
        symbols=symbols,
        initial_fraction=initial_fraction,
        max_fraction=max(0.80, initial_fraction),
        min_fraction=min(0.02, initial_fraction),
    )


def _make_orch_config(
    pod_configs: tuple[PodConfig, ...],
) -> OrchestratorConfig:
    return OrchestratorConfig(
        pods=pod_configs,
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
    )


def _mock_client() -> MagicMock:
    """BinanceClient Mock."""
    client = MagicMock()
    client.fetch_ohlcv_raw = AsyncMock(
        return_value=[
            [1704067200000, 50000, 50100, 49900, 50050, 1000],
            [1704153600000, 50050, 50200, 49950, 50100, 1100],
            [1704240000000, 50100, 50300, 50000, 50200, 1200],
        ]
    )
    return client


def _mock_futures_client() -> MagicMock:
    """BinanceFuturesClient Mock."""
    client = MagicMock()
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 100000}})
    client.fetch_open_orders = AsyncMock(return_value=[])
    client.to_futures_symbol = MagicMock(side_effect=lambda s: s.replace("/", ""))
    return client


# ── Tests ─────────────────────────────────────────────────────────


class TestOrchestratedPaperCreation:
    """orchestrated_paper classmethod 테스트."""

    @patch("src.strategy.tsmom.strategy.TSMOMStrategy.from_params")
    @patch("src.orchestrator.pod.build_pods")
    def test_orchestrated_paper_creation(
        self,
        mock_build_pods: MagicMock,
        mock_from_params: MagicMock,
    ) -> None:
        """orchestrated_paper()로 LiveRunner가 정상 생성."""
        mock_from_params.return_value = SimpleTestStrategy()
        pod_cfg = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        mock_build_pods.return_value = [
            StrategyPod(config=pod_cfg, strategy=SimpleTestStrategy(), capital_fraction=1.0)
        ]

        orch_config = _make_orch_config((pod_cfg,))
        client = _mock_client()

        runner = LiveRunner.orchestrated_paper(
            orchestrator_config=orch_config,
            target_timeframe="1D",
            client=client,
        )

        assert isinstance(runner, LiveRunner)
        assert runner.mode == LiveMode.PAPER
        assert runner._orchestrator is not None

    @patch("src.strategy.tsmom.strategy.TSMOMStrategy.from_params")
    @patch("src.orchestrator.pod.build_pods")
    def test_orchestrator_field_set(
        self,
        mock_build_pods: MagicMock,
        mock_from_params: MagicMock,
    ) -> None:
        """_orchestrator 필드가 StrategyOrchestrator로 세팅."""
        mock_from_params.return_value = SimpleTestStrategy()
        pod_cfg = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        mock_build_pods.return_value = [
            StrategyPod(config=pod_cfg, strategy=SimpleTestStrategy(), capital_fraction=1.0)
        ]

        orch_config = _make_orch_config((pod_cfg,))
        client = _mock_client()

        runner = LiveRunner.orchestrated_paper(
            orchestrator_config=orch_config,
            target_timeframe="1D",
            client=client,
        )

        assert isinstance(runner._orchestrator, StrategyOrchestrator)


class TestOrchestratedLiveCreation:
    """orchestrated_live classmethod 테스트."""

    @patch("src.eda.derivatives_feed.LiveDerivativesFeed")
    @patch("src.strategy.tsmom.strategy.TSMOMStrategy.from_params")
    @patch("src.orchestrator.pod.build_pods")
    def test_orchestrated_live_creation(
        self,
        mock_build_pods: MagicMock,
        mock_from_params: MagicMock,
        mock_deriv_feed: MagicMock,
    ) -> None:
        """orchestrated_live()로 LiveRunner가 정상 생성."""
        mock_from_params.return_value = SimpleTestStrategy()
        pod_cfg = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        mock_build_pods.return_value = [
            StrategyPod(config=pod_cfg, strategy=SimpleTestStrategy(), capital_fraction=1.0)
        ]

        orch_config = _make_orch_config((pod_cfg,))
        client = _mock_client()
        futures_client = _mock_futures_client()

        runner = LiveRunner.orchestrated_live(
            orchestrator_config=orch_config,
            target_timeframe="1D",
            client=client,
            futures_client=futures_client,
        )

        assert isinstance(runner, LiveRunner)
        assert runner.mode == LiveMode.LIVE
        assert runner._orchestrator is not None


class TestStandardModeUnaffected:
    """기존 paper/shadow/live 모드가 orchestrator 변경에 영향받지 않는지 검증."""

    def test_standard_paper_no_orchestrator(self) -> None:
        """표준 paper 모드: _orchestrator is None."""
        strategy = SimpleTestStrategy()
        client = _mock_client()

        runner = LiveRunner.paper(
            strategy=strategy,
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=PortfolioManagerConfig(),
            client=client,
        )

        assert runner._orchestrator is None

    def test_standard_shadow_no_orchestrator(self) -> None:
        """표준 shadow 모드: _orchestrator is None."""
        strategy = SimpleTestStrategy()
        client = _mock_client()

        runner = LiveRunner.shadow(
            strategy=strategy,
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=PortfolioManagerConfig(),
            client=client,
        )

        assert runner._orchestrator is None


class TestWarmupOrchestrator:
    """_warmup_orchestrator 테스트."""

    async def test_warmup_orchestrator_calls_inject(self) -> None:
        """각 pod 심볼별 inject_warmup 호출 확인."""
        strategy = SimpleTestStrategy()
        client = _mock_client()

        runner = LiveRunner.paper(
            strategy=strategy,
            symbols=["BTC/USDT"],
            target_timeframe="1D",
            config=PortfolioManagerConfig(),
            client=client,
        )

        pod_cfg = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        pod = StrategyPod(config=pod_cfg, strategy=strategy, capital_fraction=1.0)

        orch_config = _make_orch_config((pod_cfg,))
        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[pod],
            allocator=allocator,
            target_timeframe="1D",
        )

        await runner._warmup_orchestrator(orchestrator)

        # inject_warmup이 호출되었으므로 버퍼에 데이터가 있어야 함
        assert len(pod._buffers.get("BTC/USDT", [])) > 0

    async def test_warmup_orchestrator_no_client(self) -> None:
        """client가 None일 때 warmup 스킵."""
        strategy = SimpleTestStrategy()
        from src.eda.executors import BacktestExecutor
        from src.eda.live_data_feed import LiveDataFeed

        mock_client = _mock_client()
        feed = LiveDataFeed(["BTC/USDT"], "1D", mock_client)

        runner = LiveRunner(
            strategy=strategy,
            feed=feed,
            executor=BacktestExecutor(cost_model=CostModel.zero()),
            target_timeframe="1D",
            config=PortfolioManagerConfig(),
            mode=LiveMode.PAPER,
        )
        # _client is None by default
        assert runner._client is None

        pod_cfg = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod_cfg,))
        allocator = CapitalAllocator(config=orch_config)
        orchestrator = StrategyOrchestrator(
            config=orch_config,
            pods=[],
            allocator=allocator,
        )

        # Should not raise
        await runner._warmup_orchestrator(orchestrator)


class TestStartupSummary:
    """startup summary 로그 테스트."""

    @patch("src.strategy.tsmom.strategy.TSMOMStrategy.from_params")
    @patch("src.orchestrator.pod.build_pods")
    def test_startup_summary_orchestrated(
        self,
        mock_build_pods: MagicMock,
        mock_from_params: MagicMock,
    ) -> None:
        """Orchestrated 모드에서 startup summary에 pod 수가 포함."""
        mock_from_params.return_value = SimpleTestStrategy()
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 0.5)
        pod_b = _make_pod_config("pod-b", ("ETH/USDT",), 0.5)
        mock_build_pods.return_value = [
            StrategyPod(config=pod_a, strategy=SimpleTestStrategy(), capital_fraction=0.5),
            StrategyPod(config=pod_b, strategy=SimpleTestStrategy(), capital_fraction=0.5),
        ]

        orch_config = _make_orch_config((pod_a, pod_b))
        client = _mock_client()

        runner = LiveRunner.orchestrated_paper(
            orchestrator_config=orch_config,
            target_timeframe="1D",
            client=client,
        )

        # _log_startup_summary 호출이 에러 없이 완료되어야 함
        runner._log_startup_summary(100_000.0)
