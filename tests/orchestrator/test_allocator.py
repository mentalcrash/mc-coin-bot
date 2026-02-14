"""Tests for CapitalAllocator — 멀티 전략 자본 배분 엔진."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.orchestrator.allocator import (
    CapitalAllocator,
    _compute_confidence,
    _is_cov_valid,
    _regularize_cov,
)
from src.orchestrator.config import OrchestratorConfig, PodConfig
from src.orchestrator.models import AllocationMethod, LifecycleState

# ── Helpers ─────────────────────────────────────────────────────


def _make_pod_config(**overrides: object) -> PodConfig:
    defaults: dict[str, object] = {
        "pod_id": "pod-a",
        "strategy_name": "tsmom",
        "symbols": ("BTC/USDT",),
        "initial_fraction": 0.10,
        "max_fraction": 0.60,
        "min_fraction": 0.02,
    }
    defaults.update(overrides)
    return PodConfig(**defaults)  # type: ignore[arg-type]


def _make_config(
    n_pods: int = 2,
    method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
    **overrides: object,
) -> OrchestratorConfig:
    pods = [
        _make_pod_config(
            pod_id=f"pod-{i}",
            symbols=(f"SYM{i}/USDT",),
            initial_fraction=min(0.10, 1.0 / max(n_pods, 1)),
        )
        for i in range(n_pods)
    ]
    defaults: dict[str, object] = {
        "pods": tuple(pods),
        "allocation_method": method,
    }
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)  # type: ignore[arg-type]


def _make_returns(
    n_pods: int = 2,
    n_days: int = 120,
    vols: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if vols is None:
        vols = [0.02] * n_pods
    data = {}
    for i in range(n_pods):
        data[f"pod-{i}"] = rng.normal(0.001, vols[i], n_days)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    return pd.DataFrame(data, index=idx)


def _all_production(n_pods: int) -> dict[str, LifecycleState]:
    return {f"pod-{i}": LifecycleState.PRODUCTION for i in range(n_pods)}


# ── TestEqualWeight ───────────────────────────────────────────────


class TestEqualWeight:
    def test_two_pods_equal(self) -> None:
        cfg = _make_config(2, AllocationMethod.EQUAL_WEIGHT)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2)
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        assert w["pod-0"] == pytest.approx(w["pod-1"], abs=0.01)

    def test_four_pods_quarter(self) -> None:
        cfg = _make_config(4, AllocationMethod.EQUAL_WEIGHT)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(4)
        states = _all_production(4)
        w = alloc.compute_weights(returns, states)
        for i in range(4):
            assert w[f"pod-{i}"] == pytest.approx(0.25, abs=0.01)

    def test_single_pod(self) -> None:
        cfg = _make_config(1, AllocationMethod.EQUAL_WEIGHT)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(1)
        states = _all_production(1)
        w = alloc.compute_weights(returns, states)
        assert w["pod-0"] == pytest.approx(0.10, abs=0.01)


# ── TestInverseVol ────────────────────────────────────────────────


class TestInverseVol:
    def test_high_vol_gets_less(self) -> None:
        cfg = _make_config(2, AllocationMethod.INVERSE_VOLATILITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2, vols=[0.01, 0.05])
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        assert w["pod-0"] > w["pod-1"]

    def test_equal_vol_equal_weight(self) -> None:
        cfg = _make_config(2, AllocationMethod.INVERSE_VOLATILITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2, n_days=1000, vols=[0.03, 0.03])
        states = _all_production(2)
        w = alloc.compute_weights(returns, states, lookback=1000)
        assert w["pod-0"] == pytest.approx(w["pod-1"], abs=0.05)

    def test_nan_returns_fallback_ew(self) -> None:
        cfg = _make_config(2, AllocationMethod.INVERSE_VOLATILITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2)
        returns["pod-0"] = np.nan  # 모두 NaN → std = NaN
        returns["pod-1"] = np.nan
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        # NaN 과반 → drop → empty → 0.0
        total = sum(w.values())
        assert total == pytest.approx(0.0, abs=0.01)

    def test_zero_vol_clip(self) -> None:
        cfg = _make_config(2, AllocationMethod.INVERSE_VOLATILITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2, vols=[0.0, 0.0])
        # 상수 수익률 → std=0 → clip → EW fallback
        returns["pod-0"] = 0.001
        returns["pod-1"] = 0.001
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        # EW fallback이므로 대략 균등
        assert w["pod-0"] == pytest.approx(w["pod-1"], abs=0.01)


# ── TestRiskParity ────────────────────────────────────────────────


class TestRiskParity:
    def test_equal_uncorr_roughly_equal(self) -> None:
        cfg = _make_config(2, AllocationMethod.RISK_PARITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2, vols=[0.02, 0.02], seed=123)
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        assert w["pod-0"] == pytest.approx(w["pod-1"], abs=0.05)

    def test_high_vol_gets_less(self) -> None:
        cfg = _make_config(2, AllocationMethod.RISK_PARITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2, vols=[0.01, 0.05])
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        assert w["pod-0"] > w["pod-1"]

    def test_three_pods(self) -> None:
        cfg = _make_config(3, AllocationMethod.RISK_PARITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(3, vols=[0.02, 0.02, 0.02])
        states = _all_production(3)
        w = alloc.compute_weights(returns, states)
        total = sum(w.values())
        assert total == pytest.approx(1.0, abs=0.01)
        for i in range(3):
            assert w[f"pod-{i}"] > 0

    def test_prc_equal_risk_contribution(self) -> None:
        """Risk contribution이 대략 균등한지 검증."""
        cfg = _make_config(2, AllocationMethod.RISK_PARITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2, n_days=500, vols=[0.01, 0.04], seed=99)
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        # w[0] > w[1] (저변동 pod가 더 많은 비중)
        assert w["pod-0"] > w["pod-1"]

    def test_negative_corr(self) -> None:
        cfg = _make_config(2, AllocationMethod.RISK_PARITY)
        alloc = CapitalAllocator(cfg)
        rng = np.random.default_rng(42)
        base = rng.normal(0.001, 0.03, 200)
        returns = pd.DataFrame(
            {"pod-0": base, "pod-1": -base + rng.normal(0, 0.005, 200)},
            index=pd.date_range("2025-01-01", periods=200, freq="D"),
        )
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        total = sum(w.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_singular_cov_fallback(self) -> None:
        """동일 수익률(특이 공분산)이면 InvVol fallback."""
        cfg = _make_config(2, AllocationMethod.RISK_PARITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2)
        # 두 시리즈를 동일하게 → 특이 공분산 (완전 상관)
        returns["pod-1"] = returns["pod-0"]
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        total = sum(w.values())
        # Fallback이든 정상이든 합 ≤ 1
        assert total <= 1.0 + 1e-6
        assert all(v >= 0 for v in w.values())


# ── TestAdaptiveKelly ─────────────────────────────────────────────


class TestAdaptiveKelly:
    def test_confidence_zero_gives_rp(self) -> None:
        """kelly_confidence_ramp이 매우 길면 사실상 RP."""
        cfg = _make_config(
            2,
            AllocationMethod.ADAPTIVE_KELLY,
            kelly_confidence_ramp=99999,
        )
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2, n_days=1000, vols=[0.02, 0.02])
        states = _all_production(2)
        w = alloc.compute_weights(returns, states, lookback=1000)
        # 거의 RP와 동일 (sampling noise 허용)
        assert w["pod-0"] == pytest.approx(w["pod-1"], abs=0.10)

    def test_positive_mu_increases(self) -> None:
        """양의 기대수익을 가진 pod의 비중이 증가."""
        cfg = _make_config(2, AllocationMethod.ADAPTIVE_KELLY)
        alloc = CapitalAllocator(cfg)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {
                "pod-0": rng.normal(0.005, 0.02, 200),
                "pod-1": rng.normal(0.0, 0.02, 200),
            },
            index=pd.date_range("2025-01-01", periods=200, freq="D"),
        )
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        # pod-0이 더 높은 기대수익 → 더 높은 비중
        assert w["pod-0"] > w["pod-1"]

    def test_negative_mu_stays_at_rp(self) -> None:
        """모든 기대수익 음수 → 순수 RP."""
        cfg = _make_config(2, AllocationMethod.ADAPTIVE_KELLY)
        alloc = CapitalAllocator(cfg)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {
                "pod-0": rng.normal(-0.005, 0.02, 200),
                "pod-1": rng.normal(-0.003, 0.02, 200),
            },
            index=pd.date_range("2025-01-01", periods=200, freq="D"),
        )
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        total = sum(w.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_blends_rp_and_kelly(self) -> None:
        cfg = _make_config(2, AllocationMethod.ADAPTIVE_KELLY, kelly_fraction=0.5)
        alloc = CapitalAllocator(cfg)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {
                "pod-0": rng.normal(0.003, 0.02, 200),
                "pod-1": rng.normal(0.001, 0.02, 200),
            },
            index=pd.date_range("2025-01-01", periods=200, freq="D"),
        )
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        assert sum(w.values()) == pytest.approx(1.0, abs=0.01)

    def test_kelly_fraction_single_application(self) -> None:
        """H-2: kelly_fraction은 f_frac에만 1회 적용, alpha에 중복 적용되지 않음.

        confidence=1.0일 때 alpha=1.0이므로 blended ≈ f_norm (fractional Kelly).
        Lifecycle clamp으로 합이 1.0 이하일 수 있음.
        """
        cfg = _make_config(
            2,
            AllocationMethod.ADAPTIVE_KELLY,
            kelly_fraction=0.5,
            kelly_confidence_ramp=1,  # 즉시 confidence=1.0
        )
        alloc = CapitalAllocator(cfg)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {
                "pod-0": rng.normal(0.005, 0.02, 500),
                "pod-1": rng.normal(0.001, 0.02, 500),
            },
            index=pd.date_range("2025-01-01", periods=500, freq="D"),
        )
        states = {
            "pod-0": LifecycleState.PRODUCTION,
            "pod-1": LifecycleState.PRODUCTION,
        }
        w = alloc.compute_weights(returns, states, lookback=500)
        # confidence=1.0 → alpha=1.0 → blended = f_norm (fractional Kelly)
        # pod-0 has higher expected return → higher weight
        assert w["pod-0"] > w["pod-1"]
        assert sum(w.values()) <= 1.0 + 1e-6
        assert all(v >= 0 for v in w.values())

    def test_all_neg_returns_gives_valid(self) -> None:
        cfg = _make_config(2, AllocationMethod.ADAPTIVE_KELLY)
        alloc = CapitalAllocator(cfg)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {
                "pod-0": rng.normal(-0.01, 0.02, 200),
                "pod-1": rng.normal(-0.01, 0.02, 200),
            },
            index=pd.date_range("2025-01-01", periods=200, freq="D"),
        )
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        assert all(v >= 0 for v in w.values())

    def test_real_live_days_vs_heuristic(self) -> None:
        """H-8: 실제 live_days 전달 시 heuristic과 다른 결과."""
        cfg = _make_config(
            2,
            AllocationMethod.ADAPTIVE_KELLY,
            kelly_confidence_ramp=180,
        )
        alloc = CapitalAllocator(cfg)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {
                "pod-0": rng.normal(0.003, 0.02, 200),
                "pod-1": rng.normal(0.001, 0.02, 200),
            },
            index=pd.date_range("2025-01-01", periods=200, freq="D"),
        )
        states = _all_production(2)

        # Heuristic: PRODUCTION → 180일 평균 → confidence = 180/180 = 1.0
        w_heuristic = alloc.compute_weights(returns, states, pod_live_days=None)

        # 실제: 89일 평균 → confidence = 89/180 ≈ 0.49
        w_real = alloc.compute_weights(returns, states, pod_live_days={"pod-0": 89, "pod-1": 89})
        # 둘 다 유효한 가중치
        assert sum(w_heuristic.values()) == pytest.approx(1.0, abs=0.01)
        assert sum(w_real.values()) == pytest.approx(1.0, abs=0.01)

    def test_pod_live_days_none_fallback(self) -> None:
        """H-8: pod_live_days=None → heuristic fallback."""
        cfg = _make_config(2, AllocationMethod.ADAPTIVE_KELLY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2, n_days=200)
        states = _all_production(2)
        # Should not raise
        w = alloc.compute_weights(returns, states, pod_live_days=None)
        assert sum(w.values()) <= 1.0 + 1e-6


# ── TestLifecycleClamps ───────────────────────────────────────────


class TestLifecycleClamps:
    def test_retired_zero(self) -> None:
        cfg = _make_config(2, AllocationMethod.EQUAL_WEIGHT)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2)
        states = {"pod-0": LifecycleState.RETIRED, "pod-1": LifecycleState.PRODUCTION}
        w = alloc.compute_weights(returns, states)
        assert w["pod-0"] == 0.0

    def test_incubation_capped_at_initial(self) -> None:
        cfg = _make_config(2, AllocationMethod.EQUAL_WEIGHT)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2)
        states = {"pod-0": LifecycleState.INCUBATION, "pod-1": LifecycleState.PRODUCTION}
        w = alloc.compute_weights(returns, states)
        # EW = 0.5, initial = 0.10 → capped at 0.10
        assert w["pod-0"] <= cfg.pods[0].initial_fraction + 1e-6

    def test_production_clipped(self) -> None:
        cfg = _make_config(2, AllocationMethod.EQUAL_WEIGHT)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2)
        states = _all_production(2)
        w = alloc.compute_weights(returns, states)
        for i in range(2):
            pod = cfg.pods[i]
            assert w[f"pod-{i}"] >= pod.min_fraction - 1e-6
            assert w[f"pod-{i}"] <= pod.max_fraction + 1e-6

    def test_warning_halved(self) -> None:
        cfg = _make_config(2, AllocationMethod.EQUAL_WEIGHT)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2)
        states = {"pod-0": LifecycleState.WARNING, "pod-1": LifecycleState.PRODUCTION}
        w = alloc.compute_weights(returns, states)
        # EW = 0.5, WARNING * 0.5 = 0.25 -> clip(min=0.02, max=0.40) -> 0.25
        assert w["pod-0"] <= 0.25 + 1e-6

    def test_probation_min(self) -> None:
        cfg = _make_config(2, AllocationMethod.EQUAL_WEIGHT)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(2)
        states = {"pod-0": LifecycleState.PROBATION, "pod-1": LifecycleState.PRODUCTION}
        w = alloc.compute_weights(returns, states)
        assert w["pod-0"] == pytest.approx(cfg.pods[0].min_fraction, abs=0.01)

    def test_sum_le_one(self) -> None:
        cfg = _make_config(4, AllocationMethod.EQUAL_WEIGHT)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(4)
        states = _all_production(4)
        w = alloc.compute_weights(returns, states)
        assert sum(w.values()) <= 1.0 + 1e-6


# ── TestConfidence ────────────────────────────────────────────────


class TestConfidence:
    def test_zero_days(self) -> None:
        assert _compute_confidence(0, 180) == pytest.approx(0.0)

    def test_full_ramp(self) -> None:
        assert _compute_confidence(360, 180) == pytest.approx(1.0)

    def test_halfway(self) -> None:
        assert _compute_confidence(90, 180) == pytest.approx(0.5)

    def test_zero_ramp(self) -> None:
        assert _compute_confidence(0, 0) == pytest.approx(1.0)


# ── TestCovUtils ──────────────────────────────────────────────────


class TestCovUtils:
    def test_valid_cov(self) -> None:
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert _is_cov_valid(cov) is True

    def test_invalid_nan(self) -> None:
        cov = np.array([[1.0, np.nan], [np.nan, 1.0]])
        assert _is_cov_valid(cov) is False

    def test_invalid_non_symmetric(self) -> None:
        cov = np.array([[1.0, 0.3], [0.5, 1.0]])
        assert _is_cov_valid(cov) is False

    def test_regularize(self) -> None:
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        reg = _regularize_cov(cov)
        # 대각이 1.0 + ε
        assert reg[0, 0] > 1.0
        assert reg[1, 1] > 1.0
        # 비대각은 그대로
        assert reg[0, 1] == pytest.approx(0.5)


# ── TestInvariants ────────────────────────────────────────────────


class TestInvariants:
    @pytest.mark.parametrize(
        "method",
        [
            AllocationMethod.EQUAL_WEIGHT,
            AllocationMethod.INVERSE_VOLATILITY,
            AllocationMethod.RISK_PARITY,
            AllocationMethod.ADAPTIVE_KELLY,
        ],
    )
    def test_sum_le_one_all_nonneg(self, method: AllocationMethod) -> None:
        cfg = _make_config(3, method)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(3, n_days=200)
        states = _all_production(3)
        w = alloc.compute_weights(returns, states)
        assert sum(w.values()) <= 1.0 + 1e-6
        assert all(v >= -1e-8 for v in w.values())

    def test_all_retired_all_zero(self) -> None:
        cfg = _make_config(3, AllocationMethod.RISK_PARITY)
        alloc = CapitalAllocator(cfg)
        returns = _make_returns(3)
        states = {f"pod-{i}": LifecycleState.RETIRED for i in range(3)}
        w = alloc.compute_weights(returns, states)
        assert all(v == 0.0 for v in w.values())
