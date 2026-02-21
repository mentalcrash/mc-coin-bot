"""Tests for IntraPodAllocator — Pod 내 에셋 간 비중 배분 엔진."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.orchestrator.asset_allocator import (
    AssetAllocationConfig,
    IntraPodAllocator,
    _clamp_and_normalize,
)
from src.orchestrator.config import PodConfig
from src.orchestrator.models import AssetAllocationMethod
from src.orchestrator.pod import StrategyPod

# ── Helpers ─────────────────────────────────────────────────────


def _make_allocator(
    method: AssetAllocationMethod = AssetAllocationMethod.EQUAL_WEIGHT,
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"),
    **overrides: Any,
) -> IntraPodAllocator:
    defaults: dict[str, Any] = {
        "method": method,
        "vol_lookback": 20,
        "rebalance_bars": 1,
        "min_weight": 0.05,
        "max_weight": 0.60,
    }
    defaults.update(overrides)
    config = AssetAllocationConfig(**defaults)
    return IntraPodAllocator(config=config, symbols=symbols)


def _make_returns(
    symbols: tuple[str, ...],
    n: int = 60,
    vols: list[float] | None = None,
    seed: int = 42,
) -> dict[str, list[float]]:
    """심볼별 가상 수익률 생성."""
    rng = np.random.default_rng(seed)
    if vols is None:
        vols = [0.02] * len(symbols)
    result: dict[str, list[float]] = {}
    for i, s in enumerate(symbols):
        result[s] = rng.normal(0.001, vols[i], n).tolist()
    return result


def _make_pod_config(
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"),
    asset_allocation: AssetAllocationConfig | None = None,
) -> PodConfig:
    return PodConfig(
        pod_id="test-pod",
        strategy_name="tsmom",
        symbols=symbols,
        initial_fraction=0.50,
        max_fraction=0.60,
        min_fraction=0.02,
        asset_allocation=asset_allocation,
    )


def _make_strategy_mock() -> MagicMock:
    """BaseStrategy mock: run_incremental returns (df, signals_df)."""
    strategy = MagicMock()
    strategy.config = None
    return strategy


def _make_signal_result(direction: int = 1, strength: float = 0.8) -> MagicMock:
    """Mock signal result with direction and strength columns."""
    import pandas as pd

    signals = MagicMock()
    signals.direction = pd.Series([direction])
    signals.strength = pd.Series([strength])
    return signals


# ── TestAssetAllocationConfig ─────────────────────────────────────


class TestAssetAllocationConfig:
    """Config validation tests."""

    def test_defaults(self) -> None:
        config = AssetAllocationConfig()
        assert config.method == AssetAllocationMethod.EQUAL_WEIGHT
        assert config.vol_lookback == 60
        assert config.rebalance_bars == 5
        assert config.min_weight == 0.05
        assert config.max_weight == 0.60

    def test_frozen(self) -> None:
        config = AssetAllocationConfig()
        with pytest.raises(Exception):  # noqa: B017
            config.method = AssetAllocationMethod.RISK_PARITY  # type: ignore[misc]

    def test_min_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"min_weight.*cannot exceed.*max_weight"):
            AssetAllocationConfig(min_weight=0.50, max_weight=0.20)

    def test_json_roundtrip(self) -> None:
        config = AssetAllocationConfig(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            vol_lookback=30,
        )
        data = config.model_dump(mode="json")
        restored = AssetAllocationConfig.model_validate(data)
        assert restored == config


# ── TestEqualWeight ───────────────────────────────────────────────


class TestEqualWeight:
    """Equal weight allocation tests."""

    def test_equal_distribution(self) -> None:
        alloc = _make_allocator(method=AssetAllocationMethod.EQUAL_WEIGHT)
        returns = _make_returns(("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"))
        weights = alloc.on_bar(returns)
        assert all(pytest.approx(0.25, abs=0.01) == w for w in weights.values())

    def test_sum_to_one(self) -> None:
        alloc = _make_allocator(method=AssetAllocationMethod.EQUAL_WEIGHT)
        returns = _make_returns(("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"))
        weights = alloc.on_bar(returns)
        assert pytest.approx(1.0) == sum(weights.values())

    def test_stable_across_bars(self) -> None:
        alloc = _make_allocator(method=AssetAllocationMethod.EQUAL_WEIGHT)
        returns = _make_returns(("BTC/USDT", "ETH/USDT"))
        w1 = alloc.on_bar(returns)
        w2 = alloc.on_bar(returns)
        assert w1 == w2


# ── TestInverseVolatility ─────────────────────────────────────────


class TestInverseVolatility:
    """Inverse volatility allocation tests."""

    def test_high_vol_gets_lower_weight(self) -> None:
        symbols = ("BTC/USDT", "SOL/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=symbols,
            min_weight=0.01,
            max_weight=0.99,
        )
        returns = _make_returns(symbols, vols=[0.01, 0.05])
        weights = alloc.on_bar(returns)
        assert weights["BTC/USDT"] > weights["SOL/USDT"]

    def test_equal_vol_gives_equal_weight(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=symbols,
        )
        # 동일 수익률 사용 → 동일 vol 보장
        shared = [0.01, -0.02, 0.015, -0.005, 0.008] * 10
        returns = {"BTC/USDT": shared, "ETH/USDT": shared}
        weights = alloc.on_bar(returns)
        assert pytest.approx(weights["BTC/USDT"], abs=0.01) == weights["ETH/USDT"]

    def test_insufficient_data_fallback_to_ew(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=symbols,
        )
        # Only 1 data point — insufficient
        returns = {"BTC/USDT": [0.01], "ETH/USDT": [0.02]}
        weights = alloc.on_bar(returns)
        assert pytest.approx(0.5, abs=0.01) == weights["BTC/USDT"]

    def test_zero_vol_handled(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=symbols,
        )
        # Constant returns → zero variance → should use _MIN_VOL clip
        returns = {"BTC/USDT": [0.01] * 20, "ETH/USDT": [0.01] * 20}
        weights = alloc.on_bar(returns)
        assert pytest.approx(1.0) == sum(weights.values())

    def test_clamp_applied(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=symbols,
            min_weight=0.20,
            max_weight=0.80,
        )
        returns = _make_returns(symbols, vols=[0.005, 0.10])
        weights = alloc.on_bar(returns)
        assert weights["ETH/USDT"] >= 0.20
        assert weights["BTC/USDT"] <= 0.80


# ── TestRiskParity ────────────────────────────────────────────────


class TestRiskParity:
    """Risk parity allocation tests."""

    def test_high_vol_gets_lower_weight(self) -> None:
        symbols = ("BTC/USDT", "SOL/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.RISK_PARITY,
            symbols=symbols,
            min_weight=0.01,
            max_weight=0.99,
        )
        returns = _make_returns(symbols, n=100, vols=[0.01, 0.05])
        weights = alloc.on_bar(returns)
        assert weights["BTC/USDT"] > weights["SOL/USDT"]

    def test_equal_vol_gives_similar_weight(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.RISK_PARITY,
            symbols=symbols,
        )
        # Risk parity는 sample covariance 차이에 민감 → 넓은 tolerance
        returns = _make_returns(symbols, n=500, vols=[0.02, 0.02])
        weights = alloc.on_bar(returns)
        assert pytest.approx(weights["BTC/USDT"], abs=0.10) == weights["ETH/USDT"]

    def test_failure_fallback_to_iv(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.RISK_PARITY,
            symbols=symbols,
        )
        # 1 data point → insufficient → falls back to IV → falls back to EW
        returns = {"BTC/USDT": [0.01], "ETH/USDT": [0.02]}
        weights = alloc.on_bar(returns)
        assert pytest.approx(1.0) == sum(weights.values())

    def test_three_assets_sum_to_one(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.RISK_PARITY,
            symbols=symbols,
            min_weight=0.01,
            max_weight=0.99,
        )
        returns = _make_returns(symbols, n=100, vols=[0.01, 0.03, 0.06])
        weights = alloc.on_bar(returns)
        assert pytest.approx(1.0) == sum(weights.values())


# ── TestSignalWeighted ────────────────────────────────────────────


class TestSignalWeighted:
    """Signal-weighted allocation tests."""

    def test_strong_signal_gets_higher_weight(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.SIGNAL_WEIGHTED,
            symbols=symbols,
            min_weight=0.01,
            max_weight=0.99,
        )
        strengths = {"BTC/USDT": 0.9, "ETH/USDT": 0.1}
        weights = alloc.on_bar({}, strengths=strengths)
        assert weights["BTC/USDT"] > weights["ETH/USDT"]

    def test_equal_signals_give_equal_weight(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.SIGNAL_WEIGHTED,
            symbols=symbols,
        )
        strengths = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        weights = alloc.on_bar({}, strengths=strengths)
        assert pytest.approx(weights["BTC/USDT"], abs=0.01) == weights["ETH/USDT"]

    def test_zero_signals_fallback_to_ew(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.SIGNAL_WEIGHTED,
            symbols=symbols,
        )
        strengths = {"BTC/USDT": 0.0, "ETH/USDT": 0.0}
        weights = alloc.on_bar({}, strengths=strengths)
        assert pytest.approx(0.5, abs=0.01) == weights["BTC/USDT"]

    def test_none_strengths_fallback_to_ew(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.SIGNAL_WEIGHTED,
            symbols=symbols,
        )
        weights = alloc.on_bar({}, strengths=None)
        assert pytest.approx(0.5, abs=0.01) == weights["BTC/USDT"]


# ── TestClampAndNormalize ─────────────────────────────────────────


class TestClampAndNormalize:
    """Clamp and normalize utility tests."""

    def test_min_clamp(self) -> None:
        raw = {"A": 0.01, "B": 0.99}
        result = _clamp_and_normalize(raw, min_weight=0.10, max_weight=0.90)
        assert result["A"] >= 0.10

    def test_max_clamp(self) -> None:
        # 3개 에셋으로 feasible한 max constraint (max=0.50 * 3 = 1.50 > 1.0)
        raw = {"A": 0.80, "B": 0.15, "C": 0.05}
        result = _clamp_and_normalize(raw, min_weight=0.05, max_weight=0.50)
        assert result["A"] <= 0.50 + 1e-9

    def test_normalized_to_one(self) -> None:
        raw = {"A": 0.3, "B": 0.3, "C": 0.4}
        result = _clamp_and_normalize(raw, min_weight=0.10, max_weight=0.50)
        assert pytest.approx(1.0) == sum(result.values())

    def test_edge_case_all_zero(self) -> None:
        raw = {"A": 0.0, "B": 0.0}
        result = _clamp_and_normalize(raw, min_weight=0.0, max_weight=1.0)
        assert pytest.approx(1.0) == sum(result.values())


# ── TestRebalanceTiming ───────────────────────────────────────────


class TestRebalanceTiming:
    """Rebalance timing tests."""

    def test_no_rebalance_before_period(self) -> None:
        alloc = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=("BTC/USDT", "ETH/USDT"),
            rebalance_bars=5,
        )
        returns = _make_returns(("BTC/USDT", "ETH/USDT"), vols=[0.01, 0.05])
        # bars 1-4: no rebalance, initial EW
        for _ in range(4):
            weights = alloc.on_bar(returns)
        assert pytest.approx(0.5, abs=0.01) == weights["BTC/USDT"]

    def test_rebalance_at_period(self) -> None:
        alloc = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=("BTC/USDT", "ETH/USDT"),
            rebalance_bars=5,
            min_weight=0.01,
            max_weight=0.99,
        )
        returns = _make_returns(("BTC/USDT", "ETH/USDT"), vols=[0.01, 0.05])
        # bar 5: rebalance happens
        for _ in range(5):
            weights = alloc.on_bar(returns)
        assert weights["BTC/USDT"] > weights["ETH/USDT"]

    def test_bar_count_increments(self) -> None:
        alloc = _make_allocator()
        returns = _make_returns(("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"))
        alloc.on_bar(returns)
        assert alloc.bar_count == 1
        alloc.on_bar(returns)
        assert alloc.bar_count == 2

    def test_initial_weights_are_ew(self) -> None:
        symbols = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
        alloc = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=symbols,
        )
        weights = alloc.weights
        for w in weights.values():
            assert pytest.approx(1.0 / 3, abs=0.01) == w


# ── TestPodIntegration ────────────────────────────────────────────


class TestPodIntegration:
    """StrategyPod with asset allocation integration tests."""

    def test_none_config_preserves_behavior(self) -> None:
        """asset_allocation=None → 기존과 동일 동작."""
        config = _make_pod_config(
            symbols=("BTC/USDT", "ETH/USDT"),
            asset_allocation=None,
        )
        strategy = _make_strategy_mock()
        pod = StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)
        assert pod._asset_allocator is None

    def test_ew_config_per_symbol_weight(self) -> None:
        """EW 설정 시 strength * (1/N) = per-symbol weight."""
        alloc_config = AssetAllocationConfig(
            method=AssetAllocationMethod.EQUAL_WEIGHT,
            rebalance_bars=1,
        )
        config = _make_pod_config(
            symbols=("BTC/USDT", "ETH/USDT"),
            asset_allocation=alloc_config,
        )
        strategy = _make_strategy_mock()
        import pandas as pd

        signals_mock = MagicMock()
        signals_mock.direction = pd.Series([1])
        signals_mock.strength = pd.Series([0.8])
        strategy.run_incremental.return_value = (MagicMock(), signals_mock)

        pod = StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)

        # Feed enough bars for warmup (50 default)
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        bar = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0}
        for _ in range(50):
            result = pod.compute_signal(
                "BTC/USDT",
                bar,
                ts,
            )

        # Last call should return signal
        assert result is not None
        direction, strength = result
        assert direction == 1
        # EW 2 symbols: strength * (1/2) = 0.4 per symbol
        assert pytest.approx(0.4, abs=0.01) == strength

    def test_iv_adjusts_strength(self) -> None:
        """IV 설정 시 고변동성 에셋은 strength 축소."""
        alloc_config = AssetAllocationConfig(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            rebalance_bars=1,
            vol_lookback=10,
            min_weight=0.01,
            max_weight=0.99,
        )
        config = _make_pod_config(
            symbols=("BTC/USDT", "SOL/USDT"),
            asset_allocation=alloc_config,
        )
        strategy = _make_strategy_mock()
        import pandas as pd

        signals_mock = MagicMock()
        signals_mock.direction = pd.Series([1])
        signals_mock.strength = pd.Series([0.8])
        strategy.run_incremental.return_value = (MagicMock(), signals_mock)

        pod = StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)

        # Inject different volatility returns into asset_returns
        rng = np.random.default_rng(42)
        pod._asset_returns["BTC/USDT"] = rng.normal(0.0, 0.01, 30).tolist()
        pod._asset_returns["SOL/USDT"] = rng.normal(0.0, 0.05, 30).tolist()

        ts = datetime(2025, 1, 1, tzinfo=UTC)
        bar = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0}

        # Warmup BTC
        for _ in range(50):
            pod.compute_signal("BTC/USDT", bar, ts)

        btc_result = pod.compute_signal("BTC/USDT", bar, ts)

        # Warmup SOL
        for _ in range(50):
            pod.compute_signal("SOL/USDT", bar, ts)

        sol_result = pod.compute_signal("SOL/USDT", bar, ts)

        assert btc_result is not None
        assert sol_result is not None
        # BTC (low vol) should get higher adjusted strength
        assert btc_result[1] > sol_result[1]

    def test_single_symbol_pod(self) -> None:
        """단일 심볼 Pod — allocator 있어도 weight=1.0."""
        alloc_config = AssetAllocationConfig(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
        )
        config = _make_pod_config(
            symbols=("BTC/USDT",),
            asset_allocation=alloc_config,
        )
        strategy = _make_strategy_mock()
        import pandas as pd

        signals_mock = MagicMock()
        signals_mock.direction = pd.Series([1])
        signals_mock.strength = pd.Series([0.8])
        strategy.run_incremental.return_value = (MagicMock(), signals_mock)

        pod = StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)

        ts = datetime(2025, 1, 1, tzinfo=UTC)
        bar = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0}
        for _ in range(50):
            result = pod.compute_signal("BTC/USDT", bar, ts)

        assert result is not None
        # Single symbol: weight=1.0 * 1 = 1.0, so strength unchanged
        assert pytest.approx(0.8, abs=0.01) == result[1]

    def test_serialization_roundtrip(self) -> None:
        """to_dict / restore_from_dict preserves allocator state."""
        alloc_config = AssetAllocationConfig(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            rebalance_bars=1,
        )
        config = _make_pod_config(
            symbols=("BTC/USDT", "ETH/USDT"),
            asset_allocation=alloc_config,
        )
        strategy = _make_strategy_mock()
        pod = StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)

        # Set some state
        pod._asset_returns = {
            "BTC/USDT": [0.01, 0.02, -0.01],
            "ETH/USDT": [-0.02, 0.03, 0.01],
        }
        assert pod._asset_allocator is not None
        pod._asset_allocator.on_bar(pod._asset_returns)

        data = pod.to_dict()

        # Restore into new pod
        pod2 = StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)
        pod2.restore_from_dict(data)

        assert pod2._asset_returns == pod._asset_returns
        assert pod2._asset_allocator is not None
        assert pod2._asset_allocator.bar_count == pod._asset_allocator.bar_count
        assert pod2._asset_allocator.weights == pod._asset_allocator.weights

    def test_target_weights_reflect_asset_allocation(self) -> None:
        """get_target_weights()에 asset allocation이 반영되는지 확인."""
        alloc_config = AssetAllocationConfig(
            method=AssetAllocationMethod.EQUAL_WEIGHT,
            rebalance_bars=1,
        )
        config = _make_pod_config(
            symbols=("BTC/USDT", "ETH/USDT"),
            asset_allocation=alloc_config,
        )
        strategy = _make_strategy_mock()
        import pandas as pd

        signals_mock = MagicMock()
        signals_mock.direction = pd.Series([1])
        signals_mock.strength = pd.Series([0.5])
        strategy.run_incremental.return_value = (MagicMock(), signals_mock)

        pod = StrategyPod(config=config, strategy=strategy, capital_fraction=0.5)

        ts = datetime(2025, 1, 1, tzinfo=UTC)
        bar = {"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0}
        for _ in range(50):
            pod.compute_signal("BTC/USDT", bar, ts)

        tw = pod.get_target_weights()
        assert "BTC/USDT" in tw


# ── TestInvariants (parametrized) ─────────────────────────────────


_ALL_METHODS = [
    AssetAllocationMethod.EQUAL_WEIGHT,
    AssetAllocationMethod.INVERSE_VOLATILITY,
    AssetAllocationMethod.RISK_PARITY,
    AssetAllocationMethod.SIGNAL_WEIGHTED,
]

_SYMBOLS_4 = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT")


class TestInvariants:
    """Cross-method invariant tests."""

    @pytest.mark.parametrize("method", _ALL_METHODS)
    def test_weights_sum_to_one(self, method: AssetAllocationMethod) -> None:
        alloc = _make_allocator(method=method, symbols=_SYMBOLS_4)
        returns = _make_returns(_SYMBOLS_4, n=100, vols=[0.01, 0.02, 0.04, 0.03])
        strengths = {"BTC/USDT": 0.8, "ETH/USDT": 0.3, "SOL/USDT": 0.5, "BNB/USDT": 0.1}
        weights = alloc.on_bar(returns, strengths=strengths)
        assert pytest.approx(1.0) == sum(weights.values())

    @pytest.mark.parametrize("method", _ALL_METHODS)
    def test_weights_non_negative(self, method: AssetAllocationMethod) -> None:
        alloc = _make_allocator(method=method, symbols=_SYMBOLS_4)
        returns = _make_returns(_SYMBOLS_4, n=100, vols=[0.01, 0.02, 0.04, 0.03])
        strengths = {"BTC/USDT": 0.8, "ETH/USDT": 0.3, "SOL/USDT": 0.5, "BNB/USDT": 0.1}
        weights = alloc.on_bar(returns, strengths=strengths)
        assert all(w >= 0 for w in weights.values())

    @pytest.mark.parametrize("method", _ALL_METHODS)
    def test_weights_above_min(self, method: AssetAllocationMethod) -> None:
        alloc = _make_allocator(
            method=method,
            symbols=_SYMBOLS_4,
            min_weight=0.10,
            max_weight=0.50,
        )
        returns = _make_returns(_SYMBOLS_4, n=100, vols=[0.01, 0.02, 0.04, 0.03])
        strengths = {"BTC/USDT": 0.8, "ETH/USDT": 0.3, "SOL/USDT": 0.5, "BNB/USDT": 0.1}
        weights = alloc.on_bar(returns, strengths=strengths)
        # clamp+normalize 반복 수렴 한계 허용 (1e-4)
        for w in weights.values():
            assert w >= 0.10 - 1e-4

    @pytest.mark.parametrize("method", _ALL_METHODS)
    def test_weights_below_max(self, method: AssetAllocationMethod) -> None:
        alloc = _make_allocator(
            method=method,
            symbols=_SYMBOLS_4,
            min_weight=0.10,
            max_weight=0.50,
        )
        returns = _make_returns(_SYMBOLS_4, n=100, vols=[0.01, 0.02, 0.04, 0.03])
        strengths = {"BTC/USDT": 0.8, "ETH/USDT": 0.3, "SOL/USDT": 0.5, "BNB/USDT": 0.1}
        weights = alloc.on_bar(returns, strengths=strengths)
        # clamp+normalize 반복 수렴 한계 허용 (1e-2)
        for w in weights.values():
            assert w <= 0.50 + 1e-2


# ── TestPersistence ───────────────────────────────────────────────


class TestPersistence:
    """Allocator serialization tests."""

    def test_to_dict_contains_state(self) -> None:
        alloc = _make_allocator(symbols=("BTC/USDT", "ETH/USDT"))
        returns = _make_returns(("BTC/USDT", "ETH/USDT"))
        alloc.on_bar(returns)
        data = alloc.to_dict()
        assert "weights" in data
        assert "bar_count" in data
        assert data["bar_count"] == 1

    def test_restore_preserves_weights(self) -> None:
        alloc = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=("BTC/USDT", "ETH/USDT"),
        )
        returns = _make_returns(("BTC/USDT", "ETH/USDT"), vols=[0.01, 0.05])
        alloc.on_bar(returns)
        original_weights = alloc.weights

        data = alloc.to_dict()

        alloc2 = _make_allocator(
            method=AssetAllocationMethod.INVERSE_VOLATILITY,
            symbols=("BTC/USDT", "ETH/USDT"),
        )
        alloc2.restore_from_dict(data)
        assert alloc2.weights == original_weights
        assert alloc2.bar_count == alloc.bar_count

    def test_restore_with_missing_keys(self) -> None:
        alloc = _make_allocator(symbols=("BTC/USDT", "ETH/USDT"))
        initial_weights = alloc.weights
        alloc.restore_from_dict({})
        # Weights should remain initial (EW)
        assert alloc.weights == initial_weights
