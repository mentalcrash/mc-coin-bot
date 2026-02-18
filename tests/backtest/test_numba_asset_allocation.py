"""Tests for Numba-optimized Asset Allocation — Parity & Correctness.

Numba fast path (_compute_rolling_asset_weights_fast)가 Python slow path
(_compute_rolling_asset_weights)와 동일한 결과를 생성하는지 검증합니다.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestEngine,
    _numba_clamp_normalize,
    _numba_inverse_vol_weights,
    _numba_single_bar_iv,
)
from src.backtest.request import MultiAssetBacktestRequest
from src.data.market_data import MultiSymbolData
from src.orchestrator.asset_allocator import AssetAllocationConfig
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

    vols = [1.0, 2.0, 3.0, 1.5]
    drifts = [0.1, 0.05, 0.15, 0.08]

    ohlcv: dict[str, pd.DataFrame] = {}
    for i, symbol in enumerate(_SYMBOLS):
        close = 100.0 + np.cumsum(rng.normal(drifts[i], vols[i], _N_BARS))
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
def iv_config() -> AssetAllocationConfig:
    """Inverse Volatility allocation config."""
    return AssetAllocationConfig(
        method=AssetAllocationMethod.INVERSE_VOLATILITY,
        vol_lookback=20,
        rebalance_bars=5,
        min_weight=0.05,
        max_weight=0.60,
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


# ── Test: Numba IV Parity ────────────────────────────────────────


class TestNumbaIVParity:
    """Numba fast path == Python slow path 검증."""

    def test_iv_parity_synthetic(
        self,
        synthetic_multi_data: MultiSymbolData,
        iv_config: AssetAllocationConfig,
    ) -> None:
        """합성 데이터로 두 경로 결과 비교 (atol=1e-8)."""
        close_df = synthetic_multi_data.close_matrix
        engine = BacktestEngine()

        # Slow path (Python)
        slow_weights = engine._compute_rolling_asset_weights(
            close_df=close_df,
            config=iv_config,
            symbols=_SYMBOLS,
            strengths_dict=None,
        )

        # Fast path (Numba)
        fast_weights = engine._compute_rolling_asset_weights_fast(
            close_df=close_df,
            config=iv_config,
            symbols=_SYMBOLS,
            strengths_dict=None,
        )

        for s in _SYMBOLS:
            np.testing.assert_allclose(
                fast_weights[s].values,
                slow_weights[s].values,
                atol=1e-8,
                err_msg=f"Parity mismatch for symbol {s}",
            )

    def test_iv_parity_various_lookbacks(
        self,
        synthetic_multi_data: MultiSymbolData,
    ) -> None:
        """다양한 vol_lookback에서 parity 유지."""
        close_df = synthetic_multi_data.close_matrix
        engine = BacktestEngine()

        for lookback in [10, 30, 60]:
            config = AssetAllocationConfig(
                method=AssetAllocationMethod.INVERSE_VOLATILITY,
                vol_lookback=lookback,
                rebalance_bars=3,
                min_weight=0.05,
                max_weight=0.60,
            )

            slow = engine._compute_rolling_asset_weights(
                close_df=close_df,
                config=config,
                symbols=_SYMBOLS,
                strengths_dict=None,
            )
            fast = engine._compute_rolling_asset_weights_fast(
                close_df=close_df,
                config=config,
                symbols=_SYMBOLS,
                strengths_dict=None,
            )

            for s in _SYMBOLS:
                np.testing.assert_allclose(
                    fast[s].values,
                    slow[s].values,
                    atol=1e-8,
                    err_msg=f"Parity mismatch for {s}, lookback={lookback}",
                )

    def test_ew_fast_trivial(
        self,
        synthetic_multi_data: MultiSymbolData,
        ew_config: AssetAllocationConfig,
    ) -> None:
        """EW fast path: 모든 bar에서 1/N."""
        close_df = synthetic_multi_data.close_matrix
        engine = BacktestEngine()

        fast_weights = engine._compute_rolling_asset_weights_fast(
            close_df=close_df,
            config=ew_config,
            symbols=_SYMBOLS,
            strengths_dict=None,
        )

        expected = 1.0 / len(_SYMBOLS)
        for s in _SYMBOLS:
            assert all(abs(v - expected) < 1e-10 for v in fast_weights[s].values), (
                f"EW weight mismatch for {s}"
            )

    def test_rp_sw_fallback(
        self,
        synthetic_multi_data: MultiSymbolData,
    ) -> None:
        """RP/SW는 slow path fallback으로 처리."""
        close_df = synthetic_multi_data.close_matrix
        engine = BacktestEngine()

        for method in [
            AssetAllocationMethod.RISK_PARITY,
            AssetAllocationMethod.SIGNAL_WEIGHTED,
        ]:
            config = AssetAllocationConfig(
                method=method,
                vol_lookback=20,
                rebalance_bars=5,
                min_weight=0.05,
                max_weight=0.60,
            )

            slow = engine._compute_rolling_asset_weights(
                close_df=close_df,
                config=config,
                symbols=_SYMBOLS,
                strengths_dict=None,
            )
            fast = engine._compute_rolling_asset_weights_fast(
                close_df=close_df,
                config=config,
                symbols=_SYMBOLS,
                strengths_dict=None,
            )

            for s in _SYMBOLS:
                np.testing.assert_allclose(
                    fast[s].values,
                    slow[s].values,
                    atol=1e-10,
                    err_msg=f"Fallback mismatch for {s}, method={method}",
                )


# ── Test: Numba Clamp & Normalize ────────────────────────────────


class TestNumbaClampNormalize:
    """_numba_clamp_normalize 단위 테스트."""

    def test_sum_to_one(self) -> None:
        """결과 weights 합이 1.0."""
        raw = np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float64)
        result = _numba_clamp_normalize(raw, 0.05, 0.60)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_min_clamp(self) -> None:
        """최소 비중 clamp 적용."""
        raw = np.array([0.01, 0.02, 0.47, 0.50], dtype=np.float64)
        result = _numba_clamp_normalize(raw, 0.10, 0.60)
        assert all(w >= 0.10 - 1e-10 for w in result)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_max_clamp(self) -> None:
        """최대 비중 clamp 적용."""
        # 4 심볼, max=0.40 → 최소 0.60 필요 (feasible)
        raw = np.array([0.10, 0.15, 0.25, 0.50], dtype=np.float64)
        result = _numba_clamp_normalize(raw, 0.05, 0.40)
        assert all(w <= 0.40 + 1e-10 for w in result)
        assert abs(result.sum() - 1.0) < 1e-10

    def test_equal_weights_unchanged(self) -> None:
        """EW는 변경되지 않음."""
        raw = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        result = _numba_clamp_normalize(raw, 0.05, 0.60)
        np.testing.assert_allclose(result, raw, atol=1e-10)


# ── Test: Numba Inverse Vol ──────────────────────────────────────


class TestNumbaInverseVol:
    """_numba_single_bar_iv 단위 테스트."""

    def test_high_vol_lower_weight(self) -> None:
        """높은 변동성 심볼 → 낮은 weight."""
        rng = np.random.default_rng(123)
        n_bars = 100
        n_symbols = 3

        # 심볼 0: 저변동, 심볼 1: 중변동, 심볼 2: 고변동
        returns = np.column_stack(
            [
                rng.normal(0.0, 0.01, n_bars),
                rng.normal(0.0, 0.03, n_bars),
                rng.normal(0.0, 0.10, n_bars),
            ]
        )

        # bar_idx = n_bars - 1 (마지막 bar, 실제 사용 범위)
        weights = _numba_single_bar_iv(returns, n_bars - 1, 60, n_symbols)

        # 저변동 심볼이 가장 높은 weight
        assert weights[0] > weights[1] > weights[2]
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_insufficient_data_ew_fallback(self) -> None:
        """데이터 부족 시 EW fallback."""
        returns = np.full((5, 3), np.nan, dtype=np.float64)
        weights = _numba_single_bar_iv(returns, bar_idx=1, vol_lookback=60, n_symbols=3)

        expected = 1.0 / 3
        np.testing.assert_allclose(weights, [expected, expected, expected], atol=1e-10)

    def test_single_nan_symbol_ew_fallback(self) -> None:
        """하나의 심볼 데이터가 전부 NaN이면 EW."""
        n_bars = 50
        rng = np.random.default_rng(99)
        returns = np.column_stack(
            [
                rng.normal(0, 0.01, n_bars),
                np.full(n_bars, np.nan),
                rng.normal(0, 0.02, n_bars),
            ]
        )

        weights = _numba_single_bar_iv(returns, bar_idx=n_bars - 1, vol_lookback=30, n_symbols=3)
        expected = 1.0 / 3
        np.testing.assert_allclose(weights, [expected, expected, expected], atol=1e-10)


# ── Test: Full Pipeline Numba Weights ────────────────────────────


class TestNumbaFullPipeline:
    """_numba_inverse_vol_weights 전체 파이프라인 테스트."""

    def test_output_shape(self) -> None:
        """출력 shape이 (n_bars, n_symbols)."""
        rng = np.random.default_rng(42)
        n_bars, n_symbols = 100, 4
        returns = rng.normal(0, 0.02, (n_bars, n_symbols))

        result = _numba_inverse_vol_weights(
            returns,
            vol_lookback=20,
            min_weight=0.05,
            max_weight=0.60,
            rebalance_bars=5,
            n_symbols=n_symbols,
        )

        assert result.shape == (n_bars, n_symbols)

    def test_weights_sum_to_one(self) -> None:
        """매 bar마다 weight 합이 1.0."""
        rng = np.random.default_rng(42)
        n_bars, n_symbols = 100, 4
        returns = rng.normal(0, 0.02, (n_bars, n_symbols))

        result = _numba_inverse_vol_weights(
            returns,
            vol_lookback=20,
            min_weight=0.05,
            max_weight=0.60,
            rebalance_bars=5,
            n_symbols=n_symbols,
        )

        for i in range(n_bars):
            assert abs(result[i].sum() - 1.0) < 1e-8, f"Sum != 1.0 at bar {i}"

    def test_initial_bars_ew(self) -> None:
        """데이터 부족 초기 bar에서 EW."""
        rng = np.random.default_rng(42)
        n_bars, n_symbols = 100, 4
        returns = np.full((n_bars, n_symbols), np.nan, dtype=np.float64)
        # 50번째 bar부터 유효 데이터
        returns[50:] = rng.normal(0, 0.02, (50, n_symbols))

        result = _numba_inverse_vol_weights(
            returns,
            vol_lookback=20,
            min_weight=0.05,
            max_weight=0.60,
            rebalance_bars=1,
            n_symbols=n_symbols,
        )

        expected_ew = 1.0 / n_symbols
        for i in range(50):
            np.testing.assert_allclose(
                result[i],
                [expected_ew] * n_symbols,
                atol=1e-10,
                err_msg=f"Not EW at bar {i}",
            )


# ── Test: End-to-End run_multi ───────────────────────────────────


class TestEndToEnd:
    """run_multi() 전체 경로 parity."""

    def test_run_multi_fast_same_result(
        self,
        synthetic_multi_data: MultiSymbolData,
        iv_config: AssetAllocationConfig,
    ) -> None:
        """run_multi()가 fast path 사용 후에도 정상 결과."""
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
        assert pytest.approx(1.0, abs=0.05) == sum(result.config.asset_weights.values())
