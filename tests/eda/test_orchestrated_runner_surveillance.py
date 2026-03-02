"""OrchestratedRunner + BacktestSurveillanceSimulator 통합 테스트."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import MultiSymbolData
from src.eda.orchestrated_runner import OrchestratedRunner
from src.orchestrator.backtest_surveillance import BacktestSurveillanceSimulator
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod
from src.orchestrator.pod import StrategyPod
from src.orchestrator.result import OrchestratedResult
from src.orchestrator.surveillance import SurveillanceConfig
from src.orchestrator.volume_matrix import VolumeMatrix, compute_volume_matrix
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals


# ── Test Strategy ─────────────────────────────────────────────────


class _SimpleTestStrategy(BaseStrategy):
    """close > open → LONG(+1), else SHORT(-1)."""

    @property
    def name(self) -> str:
        return "test_simple_surv"

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


def _make_1m_data(
    n_days: int = 14,
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


def _make_wide_universe_data(
    seed_symbols: list[str],
    wide_symbols: list[str],
    n_days: int = 14,
) -> MultiSymbolData:
    """Seed + wide 심볼의 테스트 데이터 생성."""
    all_symbols = [*seed_symbols, *wide_symbols]
    ohlcv: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(all_symbols):
        # Wide 심볼은 낮은 base (volume 차이를 두기 위해)
        base = 50000 + i * 1000 if sym in seed_symbols else 100 + i * 10
        ohlcv[sym] = _make_1m_data(n_days=n_days, base=base, seed=42 + i)

    first = ohlcv[all_symbols[0]]
    return MultiSymbolData(
        symbols=all_symbols,
        timeframe="1m",
        start=first.index[0].to_pydatetime(),
        end=first.index[-1].to_pydatetime(),
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
        max_fraction=max(0.50, initial_fraction),
        min_fraction=min(0.02, initial_fraction),
        pinned_symbols=False,  # 동적 에셋 허용
        max_assets=5,
    )


def _make_orch_config(
    pod_configs: tuple[PodConfig, ...],
    surveillance: SurveillanceConfig | None = None,
    wide_universe_symbols: tuple[str, ...] = (),
    **overrides: object,
) -> OrchestratorConfig:
    defaults: dict[str, object] = {
        "pods": pod_configs,
        "allocation_method": AllocationMethod.EQUAL_WEIGHT,
        "rebalance_calendar_days": 1,
        "cost_bps": 4.0,
        "surveillance": surveillance,
        "wide_universe_symbols": wide_universe_symbols,
    }
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)  # type: ignore[arg-type]


def _make_pods(orch_config: OrchestratorConfig) -> list[StrategyPod]:
    """테스트용 Pod 리스트 (Registry 우회)."""
    strategy = _SimpleTestStrategy()
    return [
        StrategyPod(
            config=pod_cfg,
            strategy=strategy,
            capital_fraction=pod_cfg.initial_fraction,
        )
        for pod_cfg in orch_config.pods
    ]


# ── Tests ─────────────────────────────────────────────────────────


class TestOrchestratedRunnerSurveillance:
    """OrchestratedRunner + BacktestSurveillanceSimulator 통합 테스트."""

    @pytest.mark.asyncio()
    async def test_runner_without_surveillance(self) -> None:
        """surveillance=None 시 기존 동작 유지."""
        seed_symbols = ["BTC/USDT", "ETH/USDT"]
        data = _make_wide_universe_data(seed_symbols, [], n_days=5)
        pod = _make_pod_config("pod-a", ("BTC/USDT", "ETH/USDT"), 1.0)
        orch_config = _make_orch_config((pod,))
        pods = _make_pods(orch_config)

        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
            surveillance_simulator=None,
        )
        result = await runner.run()

        assert isinstance(result, OrchestratedResult)
        assert len(result.universe_history) == 0

    @pytest.mark.asyncio()
    async def test_runner_with_surveillance_no_change(self) -> None:
        """surveillance 활성이지만 심볼 변동 없는 경우."""
        seed_symbols = ["BTC/USDT", "ETH/USDT"]
        data = _make_wide_universe_data(seed_symbols, [], n_days=5)
        pod = _make_pod_config("pod-a", ("BTC/USDT", "ETH/USDT"), 1.0)

        surv_config = SurveillanceConfig(
            enabled=True,
            scan_interval_hours=24.0,  # 1일마다 스캔
            max_total_assets=2,
        )
        orch_config = _make_orch_config((pod,), surveillance=surv_config)
        pods = _make_pods(orch_config)

        matrix = compute_volume_matrix(data.ohlcv)
        simulator = BacktestSurveillanceSimulator(
            config=surv_config,
            volume_matrix=matrix,
            seed_symbols=set(seed_symbols),
            available_symbols=set(seed_symbols),
        )

        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
            surveillance_simulator=simulator,
        )
        result = await runner.run()

        assert isinstance(result, OrchestratedResult)
        # 변동 없으므로 universe_history는 비어있거나 added/dropped 없음
        for scan in result.universe_history:
            assert len(scan.added) > 0 or len(scan.dropped) > 0

    @pytest.mark.asyncio()
    async def test_surveillance_adds_new_symbol(self) -> None:
        """Wide universe에서 신규 심볼이 추가되는 시나리오."""
        seed_symbols = ["BTC/USDT"]
        wide_symbols = ["ETH/USDT", "SOL/USDT"]
        data = _make_wide_universe_data(seed_symbols, wide_symbols, n_days=10)
        pod = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)

        surv_config = SurveillanceConfig(
            enabled=True,
            scan_interval_hours=24.0,
            max_total_assets=3,  # 3개까지 허용
        )
        orch_config = _make_orch_config((pod,), surveillance=surv_config)
        pods = _make_pods(orch_config)

        matrix = compute_volume_matrix(data.ohlcv)
        all_symbols = set(seed_symbols + wide_symbols)
        simulator = BacktestSurveillanceSimulator(
            config=surv_config,
            volume_matrix=matrix,
            seed_symbols=set(seed_symbols),
            available_symbols=all_symbols,
        )

        runner = OrchestratedRunner.backtest(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
            initial_capital=100_000.0,
            pods=pods,
            surveillance_simulator=simulator,
        )
        result = await runner.run()

        assert isinstance(result, OrchestratedResult)
        # 최소 1번 스캔이 있어야 함 (10일 데이터 + 1일 간격)
        assert len(simulator.scan_history) >= 1

    @pytest.mark.asyncio()
    async def test_surveillance_drops_symbol(self) -> None:
        """낮은 volume 심볼이 탈락되는 시나리오."""
        all_symbols = ["HIGH", "MEDIUM", "LOW"]
        # HIGH > MEDIUM > LOW volume 순서
        dates = pd.date_range("2024-01-01", periods=14, freq="1D", tz=UTC)
        matrix = VolumeMatrix(
            daily_volume={
                "HIGH": pd.Series([10000.0] * 14, index=dates),
                "MEDIUM": pd.Series([5000.0] * 14, index=dates),
                "LOW": pd.Series([100.0] * 14, index=dates),
            }
        )

        surv_config = SurveillanceConfig(
            enabled=True,
            scan_interval_hours=168.0,
            max_total_assets=2,  # 상위 2개만
        )

        simulator = BacktestSurveillanceSimulator(
            config=surv_config,
            volume_matrix=matrix,
            seed_symbols={"HIGH", "MEDIUM", "LOW"},
            available_symbols=set(all_symbols),
        )

        result = simulator.scan_at(datetime(2024, 1, 14, tzinfo=UTC))

        # LOW가 탈락
        assert "LOW" in result.dropped
        assert "HIGH" in result.qualified_symbols
        assert "MEDIUM" in result.qualified_symbols
        assert "LOW" not in result.qualified_symbols


class TestBacktestWarmupFn:
    """_backtest_warmup_fn 단위 테스트."""

    @pytest.mark.asyncio()
    async def test_warmup_returns_bars(self) -> None:
        """warmup 함수가 올바른 bar/timestamp를 반환하는지 검증."""
        seed_symbols = ["BTC/USDT"]
        data = _make_wide_universe_data(seed_symbols, [], n_days=5)
        pod = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod,))

        runner = OrchestratedRunner(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
        )
        # replay ts 설정 (3일차)
        runner._current_replay_ts = pd.Timestamp(  # type: ignore[assignment]
            datetime(2024, 1, 3, tzinfo=UTC)
        )

        bars, timestamps = await runner._backtest_warmup_fn("BTC/USDT", "1D", 2)

        assert len(bars) == 2
        assert len(timestamps) == 2
        assert all(
            set(b.keys()) == {"open", "high", "low", "close", "volume"} for b in bars
        )
        # 모든 timestamp가 replay ts 이전
        for ts in timestamps:
            assert ts <= datetime(2024, 1, 3, tzinfo=UTC)

    @pytest.mark.asyncio()
    async def test_warmup_missing_symbol(self) -> None:
        """존재하지 않는 심볼에 대해 빈 결과 반환."""
        data = _make_wide_universe_data(["BTC/USDT"], [], n_days=3)
        pod = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod,))

        runner = OrchestratedRunner(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
        )
        runner._current_replay_ts = pd.Timestamp(  # type: ignore[assignment]
            datetime(2024, 1, 3, tzinfo=UTC)
        )

        bars, timestamps = await runner._backtest_warmup_fn("NONEXIST", "1D", 10)

        assert bars == []
        assert timestamps == []

    @pytest.mark.asyncio()
    async def test_warmup_no_replay_ts(self) -> None:
        """replay_ts 미설정 시 빈 결과."""
        data = _make_wide_universe_data(["BTC/USDT"], [], n_days=3)
        pod = _make_pod_config("pod-a", ("BTC/USDT",), 1.0)
        orch_config = _make_orch_config((pod,))

        runner = OrchestratedRunner(
            orchestrator_config=orch_config,
            data=data,
            target_timeframe="1D",
        )
        # _current_replay_ts is None (default)

        bars, timestamps = await runner._backtest_warmup_fn("BTC/USDT", "1D", 2)

        assert bars == []
        assert timestamps == []


class TestPMUpdateAssetWeights:
    """PM.update_asset_weights 단위 테스트."""

    def test_update_weights(self) -> None:
        """asset_weights 갱신 + batch_mode 갱신."""
        from src.eda.portfolio_manager import EDAPortfolioManager
        from src.portfolio.config import PortfolioManagerConfig

        pm_config = PortfolioManagerConfig()
        pm = EDAPortfolioManager(
            config=pm_config,
            initial_capital=100_000.0,
            asset_weights={"BTC": 1.0},
        )

        assert pm._batch_mode is False  # 1개면 single mode

        pm.update_asset_weights({"BTC": 1.0, "ETH": 1.0})
        assert pm._asset_weights == {"BTC": 1.0, "ETH": 1.0}
        assert pm._batch_mode is True  # 2개 이상이면 batch mode
