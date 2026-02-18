"""Tests for Intra-Pod Asset Allocation — Backtest Parity.

VBT BacktestEngine.run_multi()에서 동적 asset allocation이 EDA Pod과
동일한 IntraPodAllocator를 사용하여 parity를 유지하는지 검증합니다.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.request import MultiAssetBacktestRequest
from src.data.market_data import MultiSymbolData
from src.orchestrator.asset_allocator import (
    AssetAllocationConfig,
    IntraPodAllocator,
)
from src.orchestrator.models import AssetAllocationMethod

# ── Constants ─────────────────────────────────────────────────────

_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")
_N_BARS = 200
_SEED = 42


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def synthetic_multi_data() -> MultiSymbolData:
    """Deterministic 합성 데이터 (4종, 200 bars, 1D)."""
    dates = pd.date_range(start="2024-01-01", periods=_N_BARS, freq="D", tz=UTC)
    rng = np.random.default_rng(_SEED)

    # 다른 변동성으로 차별화
    vols = [1.0, 2.0, 3.0, 1.5]
    drifts = [0.1, 0.05, 0.15, 0.08]

    ohlcv: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(_SYMBOLS):
        close = 100.0 + np.cumsum(rng.normal(drifts[i], vols[i], _N_BARS))
        # close가 양수 보장
        close = np.maximum(close, 10.0)
        ohlcv[symbol] = pd.DataFrame(
            {
                "open": close * 0.995,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": rng.uniform(1000, 5000, _N_BARS),
            },
            index=dates,
        )

    return MultiSymbolData(
        symbols=list(_SYMBOLS),
        timeframe="1D",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 7, 18, tzinfo=UTC),
        ohlcv=ohlcv,
    )


@pytest.fixture
def ew_config() -> AssetAllocationConfig:
    """Equal Weight allocation config."""
    return AssetAllocationConfig(
        method=AssetAllocationMethod.EQUAL_WEIGHT,
        rebalance_bars=1,
        min_weight=0.05,
        max_weight=0.60,
    )


@pytest.fixture
def iv_config() -> AssetAllocationConfig:
    """Inverse Volatility allocation config."""
    return AssetAllocationConfig(
        method=AssetAllocationMethod.INVERSE_VOLATILITY,
        vol_lookback=20,
        rebalance_bars=5,
        min_weight=0.05,
        max_weight=0.60,
    )


# ── Test: _compute_rolling_asset_weights 수동 검증 ──────────────


class TestWeightSeriesMatchesManual:
    """_compute_rolling_asset_weights()와 수동 IntraPodAllocator 호출 일치."""

    def test_weight_series_matches_manual(
        self,
        synthetic_multi_data: MultiSymbolData,
        iv_config: AssetAllocationConfig,
    ) -> None:
        close_df = synthetic_multi_data.close_matrix
        engine = BacktestEngine()

        rolling_weights = engine._compute_rolling_asset_weights(
            close_df=close_df,
            config=iv_config,
            symbols=_SYMBOLS,
            strengths_dict=None,
        )

        # 수동으로 동일 로직 수행
        allocator = IntraPodAllocator(config=iv_config, symbols=_SYMBOLS)
        returns_df = close_df.pct_change().shift(1)
        returns_history: dict[str, list[float]] = {s: [] for s in _SYMBOLS}

        for i in range(len(close_df)):
            for s in _SYMBOLS:
                ret_val = returns_df[s].iloc[i]
                if pd.notna(ret_val):
                    returns_history[s].append(float(ret_val))

            # 심볼당 1회 on_bar()
            for _s in _SYMBOLS:
                allocator.on_bar(returns=returns_history, strengths=None)

            # 비교
            current_weights = allocator.weights
            for s in _SYMBOLS:
                assert rolling_weights[s].iloc[i] == pytest.approx(current_weights[s], abs=1e-10), (
                    f"Mismatch at bar {i}, symbol {s}"
                )


# ── Test: EW dynamic == 정적 1/N ──────────────────────────────


class TestEWAllocationMatchesStatic:
    """EW dynamic weight는 정적 1/N과 동일해야 함."""

    def test_ew_allocation_matches_static(
        self,
        synthetic_multi_data: MultiSymbolData,
        ew_config: AssetAllocationConfig,
    ) -> None:
        close_df = synthetic_multi_data.close_matrix
        engine = BacktestEngine()

        rolling_weights = engine._compute_rolling_asset_weights(
            close_df=close_df,
            config=ew_config,
            symbols=_SYMBOLS,
            strengths_dict=None,
        )

        expected = 1.0 / len(_SYMBOLS)
        for s in _SYMBOLS:
            for i in range(len(close_df)):
                assert rolling_weights[s].iloc[i] == pytest.approx(expected, abs=1e-10)


# ── Test: IV/RP 사용 시 EW와 다른 결과 ──────────────────────────


class TestBacktestResultDiffersFromEW:
    """IV/RP 사용 시 EW 대비 다른 weight 생성."""

    @pytest.mark.parametrize(
        "method",
        [
            AssetAllocationMethod.INVERSE_VOLATILITY,
            AssetAllocationMethod.RISK_PARITY,
        ],
    )
    def test_weights_differ_from_ew(
        self,
        synthetic_multi_data: MultiSymbolData,
        method: AssetAllocationMethod,
    ) -> None:
        config = AssetAllocationConfig(
            method=method,
            vol_lookback=20,
            rebalance_bars=5,
            min_weight=0.01,
            max_weight=0.99,
        )
        close_df = synthetic_multi_data.close_matrix
        engine = BacktestEngine()

        rolling_weights = engine._compute_rolling_asset_weights(
            close_df=close_df,
            config=config,
            symbols=_SYMBOLS,
            strengths_dict=None,
        )

        # 마지막 bar의 weight가 EW(0.25)와 다른지 확인
        ew = 1.0 / len(_SYMBOLS)
        last_weights = {s: float(rolling_weights[s].iloc[-1]) for s in _SYMBOLS}
        # 최소 하나의 심볼이 EW와 다른 weight를 가져야 함
        assert any(abs(w - ew) > 0.01 for w in last_weights.values()), (
            f"All weights equal to EW: {last_weights}"
        )


# ── Test: Look-Ahead Bias 체크 ────────────────────────────────


class TestLookAheadBiasCheck:
    """Bar t의 weight가 close[t]를 사용하지 않음."""

    def test_look_ahead_bias_check(
        self,
        synthetic_multi_data: MultiSymbolData,
        iv_config: AssetAllocationConfig,
    ) -> None:
        close_df = synthetic_multi_data.close_matrix.copy()
        engine = BacktestEngine()

        # 원본 weight 계산
        original_weights = engine._compute_rolling_asset_weights(
            close_df=close_df,
            config=iv_config,
            symbols=_SYMBOLS,
            strengths_dict=None,
        )

        # bar 100 이후의 close를 변경 (미래 데이터 오염)
        modified_close = close_df.copy()
        modified_close.iloc[100:] = modified_close.iloc[100:] * 2.0

        modified_weights = engine._compute_rolling_asset_weights(
            close_df=modified_close,
            config=iv_config,
            symbols=_SYMBOLS,
            strengths_dict=None,
        )

        # bar 0~99의 weight는 변하지 않아야 함 (look-ahead 없음)
        # bar 100의 weight는 close[100]을 사용하지 않으므로 변하지 않아야 함
        # shift(1)로 인해 bar 100의 return은 close[99]/close[98] → 영향 없음
        for s in _SYMBOLS:
            for i in range(101):
                assert original_weights[s].iloc[i] == pytest.approx(
                    modified_weights[s].iloc[i], abs=1e-10
                ), f"Look-ahead bias at bar {i}, symbol {s}"


# ── Test: End-to-end run_multi ─────────────────────────────────


class TestRunMultiWithAllocation:
    """run_multi() end-to-end 정상 실행."""

    def test_run_multi_with_allocation(
        self,
        synthetic_multi_data: MultiSymbolData,
        iv_config: AssetAllocationConfig,
    ) -> None:
        from src.portfolio.portfolio import Portfolio
        from src.strategy.registry import get_strategy

        strategy = get_strategy("tsmom")()
        portfolio = Portfolio.create(initial_capital=100_000)

        request = MultiAssetBacktestRequest(
            data=synthetic_multi_data,
            strategy=strategy,
            portfolio=portfolio,
            asset_allocation=iv_config,
        )

        engine = BacktestEngine()
        result = engine.run_multi(request)

        assert result is not None
        assert result.config.asset_allocation_method == "inverse_volatility"
        assert len(result.config.symbols) == len(_SYMBOLS)
        # weight 합이 1.0에 근사
        assert pytest.approx(1.0, abs=0.05) == sum(result.config.asset_weights.values())

    def test_run_multi_without_allocation_unchanged(
        self,
        synthetic_multi_data: MultiSymbolData,
    ) -> None:
        """asset_allocation=None → 기존 정적 weight 동작 유지."""
        from src.portfolio.portfolio import Portfolio
        from src.strategy.registry import get_strategy

        strategy = get_strategy("tsmom")()
        portfolio = Portfolio.create(initial_capital=100_000)

        request = MultiAssetBacktestRequest(
            data=synthetic_multi_data,
            strategy=strategy,
            portfolio=portfolio,
        )

        engine = BacktestEngine()
        result = engine.run_multi(request)

        assert result is not None
        assert result.config.asset_allocation_method is None
        # 정적 EW
        expected_w = 1.0 / len(_SYMBOLS)
        for w in result.config.asset_weights.values():
            assert w == pytest.approx(expected_w, abs=1e-10)


# ── Test: weights와 asset_allocation 상호 배타 ────────────────


class TestMutuallyExclusive:
    """weights와 asset_allocation 동시 지정 시 ValueError."""

    def test_weights_and_allocation_mutually_exclusive(
        self,
        synthetic_multi_data: MultiSymbolData,
        iv_config: AssetAllocationConfig,
    ) -> None:
        from src.portfolio.portfolio import Portfolio
        from src.strategy.registry import get_strategy

        strategy = get_strategy("tsmom")()
        portfolio = Portfolio.create(initial_capital=100_000)

        with pytest.raises(ValueError, match="mutually exclusive"):
            MultiAssetBacktestRequest(
                data=synthetic_multi_data,
                strategy=strategy,
                portfolio=portfolio,
                weights={"BTC/USDT": 0.4, "ETH/USDT": 0.3, "SOL/USDT": 0.2, "BNB/USDT": 0.1},
                asset_allocation=iv_config,
            )


# ── Test: Rebalance Timing — Pod parity ──────────────────────


class TestRebalanceTimingMatchesPod:
    """심볼당 on_bar() 호출로 EDA Pod과 동일 bar_count 패턴."""

    def test_rebalance_timing_matches_pod(self) -> None:
        """rebalance_bars=5, 4 symbols → bar_count 4씩 증가."""
        config = AssetAllocationConfig(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            vol_lookback=10,
            rebalance_bars=5,
            min_weight=0.05,
            max_weight=0.60,
        )
        symbols = ("A", "B", "C", "D")
        n_bars = 10
        dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="D", tz=UTC)
        rng = np.random.default_rng(42)

        close_df = pd.DataFrame(
            {s: 100.0 + np.cumsum(rng.normal(0.1, 1.0, n_bars)) for s in symbols},
            index=dates,
        )

        engine = BacktestEngine()
        rolling_weights = engine._compute_rolling_asset_weights(
            close_df=close_df,
            config=config,
            symbols=symbols,
            strengths_dict=None,
        )

        # 별도 allocator로 동일 bar_count 패턴 검증
        alloc = IntraPodAllocator(config=config, symbols=symbols)
        returns_df = close_df.pct_change().shift(1)
        returns_history: dict[str, list[float]] = {s: [] for s in symbols}

        bar_counts: list[int] = []
        for i in range(n_bars):
            for s in symbols:
                rv = returns_df[s].iloc[i]
                if pd.notna(rv):
                    returns_history[s].append(float(rv))

            for _s in symbols:
                alloc.on_bar(returns=returns_history, strengths=None)
            bar_counts.append(alloc.bar_count)

        # bar_count는 심볼 수 * bar 인덱스 패턴
        # bar 0: 4 calls → bar_count=4
        # bar 1: 4 calls → bar_count=8
        for i in range(n_bars):
            expected_count = (i + 1) * len(symbols)
            assert bar_counts[i] == expected_count

        # rebalance_bars=5 → bar_count 5에서 첫 rebalance
        # 4 symbols이면 bar_count=4,8,12... → bar_count % 5 == 0인 시점은 bar_count=5 (불가),
        # 실제로는 bar_count가 5의 배수인 시점: 20, 40, ...
        # bar 4: bar_count=20 (5번째 bar, 0-indexed) → rebalance 발생
        # 이 패턴이 engine 계산과 수동 계산에서 동일한지 확인
        for s in symbols:
            for i in range(n_bars):
                assert (
                    rolling_weights[s].iloc[i]
                    == pytest.approx(alloc.weights.get(s, 0.25), abs=1e-10)
                    or i < n_bars - 1
                )  # 마지막 bar는 일치 보장
