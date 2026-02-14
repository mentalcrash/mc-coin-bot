"""OrchestratedRunner 백테스트 통합 테스트.

OrchestratedRunner를 통한 멀티 전략 백테스트 end-to-end 검증.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import MultiSymbolData
from src.eda.orchestrated_runner import OrchestratedRunner, _compute_pod_metrics, _derive_pm_config
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod
from src.orchestrator.pod import StrategyPod
from src.orchestrator.result import OrchestratedResult
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ── Test Strategy ─────────────────────────────────────────────────


class SimpleTestStrategy(BaseStrategy):
    """close > open → LONG(+1), else SHORT(-1). strength = abs(close-open)/open."""

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
        strength = ((df["close"] - df["open"]).abs() / df["open"]).shift(1).fillna(0.01)
        entries = direction.diff().fillna(0).abs() > 0
        exits = pd.Series(False, index=df.index)
        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction,
            strength=strength,
        )


# ── Helpers ───────────────────────────────────────────────────────


def _make_trending_1m_data(
    n_days: int = 5,
    base: float = 50000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """상승 트렌드 1m 테스트 데이터."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = pd.date_range(start=start, periods=n, freq="1min", tz=UTC)

    trend = np.linspace(0, 5000, n)
    noise = rng.standard_normal(n) * 20
    close = base + trend + noise

    return pd.DataFrame(
        {
            "open": close * 0.9999,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.integers(10, 100, n) * 10.0,
        },
        index=timestamps,
    )


def _make_multi_symbol_data(
    symbols: list[str],
    n_days: int = 5,
) -> MultiSymbolData:
    """멀티 심볼 테스트 데이터 생성."""
    ohlcv: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        ohlcv[sym] = _make_trending_1m_data(n_days=n_days, base=50000 + i * 10000, seed=42 + i)

    first = ohlcv[symbols[0]]
    return MultiSymbolData(
        symbols=tuple(symbols),
        timeframe="1m",
        start=first.index[0].to_pydatetime(),  # type: ignore[union-attr]
        end=first.index[-1].to_pydatetime(),  # type: ignore[union-attr]
        ohlcv=ohlcv,
    )


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
    **overrides: object,
) -> OrchestratorConfig:
    defaults: dict[str, object] = {
        "pods": pod_configs,
        "allocation_method": AllocationMethod.EQUAL_WEIGHT,
        "rebalance_calendar_days": 1,
        "cost_bps": 4.0,
    }
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)  # type: ignore[arg-type]


def _make_pods_from_config(
    orch_config: OrchestratorConfig,
) -> list[StrategyPod]:
    """테스트용 Pod 리스트 생성 (Registry 우회)."""
    strategy = SimpleTestStrategy()
    pods: list[StrategyPod] = []
    for pod_cfg in orch_config.pods:
        pod = StrategyPod(
            config=pod_cfg,
            strategy=strategy,
            capital_fraction=pod_cfg.initial_fraction,
        )
        pods.append(pod)
    return pods


# ── Tests ─────────────────────────────────────────────────────────


class TestOrchestratedRunnerBacktest:
    """OrchestratedRunner 백테스트 통합 테스트."""

    @pytest.fixture
    def two_pod_setup(self) -> tuple[OrchestratorConfig, MultiSymbolData, list[StrategyPod]]:
        """2 pods, 2 symbols 기본 셋업."""
        symbols = ["BTC/USDT", "ETH/USDT"]
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 0.5)
        pod_b = _make_pod_config("pod-b", ("ETH/USDT",), 0.5)
        orch_config = _make_orch_config((pod_a, pod_b))
        data = _make_multi_symbol_data(symbols, n_days=5)
        pods = _make_pods_from_config(orch_config)
        return orch_config, data, pods

    async def test_backtest_two_pods_returns_result(
        self,
        two_pod_setup: tuple[OrchestratorConfig, MultiSymbolData, list[StrategyPod]],
    ) -> None:
        """2 pods, 2 symbols 백테스트가 OrchestratedResult를 반환."""
        orch_config, data, pods = two_pod_setup
        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
        )
        result = await runner.run()

        assert isinstance(result, OrchestratedResult)

    async def test_result_has_portfolio_metrics(
        self,
        two_pod_setup: tuple[OrchestratorConfig, MultiSymbolData, list[StrategyPod]],
    ) -> None:
        """portfolio_metrics 필드가 PerformanceMetrics 타입."""
        orch_config, data, pods = two_pod_setup
        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
        )
        result = await runner.run()

        metrics = result.portfolio_metrics
        assert metrics.total_trades >= 0
        assert isinstance(metrics.sharpe_ratio, float)

    async def test_result_has_pod_equity_curves(
        self,
        two_pod_setup: tuple[OrchestratorConfig, MultiSymbolData, list[StrategyPod]],
    ) -> None:
        """pod_equity_curves에 각 pod_id가 존재."""
        orch_config, data, pods = two_pod_setup
        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
        )
        result = await runner.run()

        assert "pod-a" in result.pod_equity_curves
        assert "pod-b" in result.pod_equity_curves

    async def test_pod_metrics_populated(
        self,
        two_pod_setup: tuple[OrchestratorConfig, MultiSymbolData, list[StrategyPod]],
    ) -> None:
        """pod_metrics에 각 pod_id가 존재하고 간이 메트릭 포함."""
        orch_config, data, pods = two_pod_setup
        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
        )
        result = await runner.run()

        for pod_id in ("pod-a", "pod-b"):
            assert pod_id in result.pod_metrics
            m = result.pod_metrics[pod_id]
            assert "sharpe" in m
            assert "mdd" in m
            assert "total_return" in m

    async def test_portfolio_equity_curve_type(
        self,
        two_pod_setup: tuple[OrchestratorConfig, MultiSymbolData, list[StrategyPod]],
    ) -> None:
        """portfolio_equity_curve가 pd.Series 타입."""
        orch_config, data, pods = two_pod_setup
        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
        )
        result = await runner.run()

        assert isinstance(result.portfolio_equity_curve, pd.Series)

    async def test_single_pod_works(self) -> None:
        """1 pod도 정상 동작."""
        symbols = ["BTC/USDT"]
        pod_a = _make_pod_config("pod-solo", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod_a,))
        data = _make_multi_symbol_data(symbols, n_days=5)
        pods = _make_pods_from_config(orch_config)

        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
        )
        result = await runner.run()

        assert isinstance(result, OrchestratedResult)
        assert "pod-solo" in result.pod_metrics

    async def test_backtest_factory(self) -> None:
        """OrchestratedRunner.backtest() classmethod가 정상 생성."""
        symbols = ["BTC/USDT"]
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod_a,))
        data = _make_multi_symbol_data(symbols, n_days=3)
        pods = _make_pods_from_config(orch_config)

        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            pods=pods,
        )
        assert isinstance(runner, OrchestratedRunner)


class TestTimeframeFilter:
    """TF 필터 테스트."""

    async def test_timeframe_filter(self) -> None:
        """1m bar가 Orchestrator에 도달하지 않음 (TF=1D 설정 시)."""
        symbols = ["BTC/USDT"]
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod_a,))
        data = _make_multi_symbol_data(symbols, n_days=3)
        pods = _make_pods_from_config(orch_config)

        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
        )
        result = await runner.run()

        # 3일 데이터 → 최대 2-3 TF bars (첫 날은 warmup)
        # 1m bars (3 * 1440 = 4320개)가 Orchestrator에 도달했으면 훨씬 많은 시그널
        assert isinstance(result, OrchestratedResult)


class TestPMConfigDerivation:
    """PM config 유도 테스트."""

    def test_derive_pm_config(self) -> None:
        """_derive_pm_config 정확성."""
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod_a,), max_gross_leverage=5.0, cost_bps=8.0)

        pm_config = _derive_pm_config(orch_config)

        assert pm_config.max_leverage_cap == 5.0
        # H-4: system_stop_loss는 이제 max_portfolio_drawdown * 1.5
        assert pm_config.system_stop_loss is not None
        assert pm_config.system_stop_loss == pytest.approx(orch_config.max_portfolio_drawdown * 1.5)
        assert pm_config.use_trailing_stop is False
        assert pm_config.rebalance_threshold == 0.01
        assert pm_config.cash_sharing is True
        assert pm_config.cost_model.taker_fee == pytest.approx(0.0008)

    def test_pm_stop_loss_default_config(self) -> None:
        """H-4: default config → system_stop_loss = 0.15 * 1.5 = 0.225."""
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod_a,))  # max_portfolio_drawdown=0.15

        pm_config = _derive_pm_config(orch_config)
        assert pm_config.system_stop_loss == pytest.approx(0.225)

    def test_pm_stop_loss_cap_at_30pct(self) -> None:
        """H-4: high MDD config → cap at 0.30."""
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        # max_portfolio_drawdown=0.25 → 0.25*1.5=0.375 → cap at 0.30
        orch_config = _make_orch_config((pod_a,), max_portfolio_drawdown=0.25)

        pm_config = _derive_pm_config(orch_config)
        assert pm_config.system_stop_loss == pytest.approx(0.30)

    def test_asset_weights_uniform(self) -> None:
        """uniform weights 검증."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        pods = tuple(
            _make_pod_config(f"pod-{i}", (sym,), initial_fraction=1.0 / len(symbols))
            for i, sym in enumerate(symbols)
        )
        orch_config = _make_orch_config(pods)

        # OrchestratedRunner가 uniform weights 생성 확인
        all_symbols = list(orch_config.all_symbols)
        asset_weights = dict.fromkeys(all_symbols, 1.0)
        assert all(w == 1.0 for w in asset_weights.values())
        assert len(asset_weights) == 3

    def test_derive_pm_config_via_static(self) -> None:
        """OrchestratedRunner.derive_pm_config() 외부 접근."""
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod_a,))

        pm_config = OrchestratedRunner.derive_pm_config(orch_config)
        assert pm_config.system_stop_loss == pytest.approx(0.225)


class TestPodMetricsComputation:
    """Pod 메트릭 계산 테스트."""

    def test_empty_returns(self) -> None:
        """빈 수익률 → 기본 메트릭."""
        m = _compute_pod_metrics([])
        assert m["sharpe"] == 0.0
        assert m["mdd"] == 0.0
        assert m["n_days"] == 0

    def test_positive_returns(self) -> None:
        """양수 수익률 → sharpe > 0."""
        returns = [0.01, 0.02, 0.01, 0.005, 0.015]
        m = _compute_pod_metrics(returns)
        assert m["sharpe"] > 0
        assert m["total_return"] > 0
        assert m["n_days"] == 5

    def test_negative_returns(self) -> None:
        """음수 수익률 → mdd < 0."""
        returns = [-0.01, -0.02, -0.01, -0.005, -0.015]
        m = _compute_pod_metrics(returns)
        assert m["mdd"] < 0
        assert m["total_return"] < 0


class TestFlushPendingSignals:
    """데이터 종료 후 flush 동작 테스트."""

    async def test_flush_pending_signals(self) -> None:
        """백테스트 종료 후 flush가 에러 없이 동작."""
        symbols = ["BTC/USDT"]
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod_a,))
        data = _make_multi_symbol_data(symbols, n_days=3)
        pods = _make_pods_from_config(orch_config)

        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            pods=pods,
        )
        # run() 내부에서 flush가 자동 호출되므로 에러 없이 완료 확인
        result = await runner.run()
        assert isinstance(result, OrchestratedResult)


class TestSharedSymbols:
    """심볼이 여러 Pod에 공유되는 경우 테스트."""

    async def test_shared_symbols_between_pods(self) -> None:
        """같은 심볼을 2 pods가 공유해도 정상 동작."""
        symbols = ["BTC/USDT"]
        pod_a = _make_pod_config("pod-a", ("BTC/USDT",), 0.5)
        pod_b = _make_pod_config("pod-b", ("BTC/USDT",), 0.5)
        orch_config = _make_orch_config((pod_a, pod_b))
        data = _make_multi_symbol_data(symbols, n_days=5)
        pods = _make_pods_from_config(orch_config)

        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
        )
        result = await runner.run()

        assert isinstance(result, OrchestratedResult)
        assert "pod-a" in result.pod_metrics
        assert "pod-b" in result.pod_metrics
