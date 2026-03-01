"""Tests for Volatility Targeting Overlay."""

from __future__ import annotations

import numpy as np

from src.orchestrator.vol_targeting import (
    apply_vol_targeting,
    compute_realized_vol,
    compute_vol_scalar,
)

# ── Constants ─────────────────────────────────────────────────────

_TOLERANCE = 1e-6


# ── compute_realized_vol ──────────────────────────────────────────


class TestComputeRealizedVol:
    """compute_realized_vol 단위 테스트."""

    def test_basic_vol_computation(self) -> None:
        """정상 케이스: 기본 변동성 계산."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100).tolist()
        pod_returns = {"pod-a": returns, "pod-b": returns}
        weights = {"pod-a": 0.5, "pod-b": 0.5}

        vol = compute_realized_vol(pod_returns, weights, lookback=60)
        assert vol > 0.0
        # 연간화 확인: ~0.01 * sqrt(365) ≈ 0.19
        assert 0.05 < vol < 0.50

    def test_empty_pod_returns(self) -> None:
        """빈 데이터 → 0.0 반환."""
        vol = compute_realized_vol({}, {"pod-a": 0.5}, lookback=60)
        assert vol == 0.0

    def test_zero_weights(self) -> None:
        """weight가 0이면 해당 pod 제외."""
        pod_returns = {"pod-a": [0.01] * 20}
        weights = {"pod-a": 0.0}

        vol = compute_realized_vol(pod_returns, weights, lookback=60)
        assert vol == 0.0

    def test_insufficient_data(self) -> None:
        """데이터가 최소 요구보다 적으면 0.0."""
        pod_returns = {"pod-a": [0.01, 0.02, 0.03]}
        weights = {"pod-a": 1.0}

        vol = compute_realized_vol(pod_returns, weights, lookback=60)
        assert vol == 0.0

    def test_lookback_clipping(self) -> None:
        """lookback이 데이터 길이보다 크면 전체 데이터 사용."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 30).tolist()
        pod_returns = {"pod-a": returns}
        weights = {"pod-a": 1.0}

        vol = compute_realized_vol(pod_returns, weights, lookback=1000)
        assert vol > 0.0


# ── compute_vol_scalar ────────────────────────────────────────────


class TestComputeVolScalar:
    """compute_vol_scalar 단위 테스트."""

    def test_normal_scaling(self) -> None:
        """target=15%, realized=30% → scalar=0.5."""
        scalar = compute_vol_scalar(0.30, 0.15, floor=0.10, cap=1.5)
        assert abs(scalar - 0.5) < _TOLERANCE

    def test_low_vol_caps(self) -> None:
        """실현 변동성이 낮으면 cap에 클램프."""
        scalar = compute_vol_scalar(0.05, 0.15, floor=0.10, cap=1.5)
        assert abs(scalar - 1.5) < _TOLERANCE

    def test_high_vol_floors(self) -> None:
        """실현 변동성이 매우 높으면 floor에 클램프."""
        scalar = compute_vol_scalar(1.0, 0.15, floor=0.10, cap=1.5)
        assert abs(scalar - 0.15) < _TOLERANCE

    def test_zero_realized_vol(self) -> None:
        """realized_vol=0 → cap 반환."""
        scalar = compute_vol_scalar(0.0, 0.15, floor=0.10, cap=1.5)
        assert abs(scalar - 1.5) < _TOLERANCE

    def test_target_equals_realized(self) -> None:
        """target = realized → scalar = 1.0."""
        scalar = compute_vol_scalar(0.15, 0.15, floor=0.10, cap=1.5)
        assert abs(scalar - 1.0) < _TOLERANCE


# ── apply_vol_targeting ───────────────────────────────────────────


class TestApplyVolTargeting:
    """apply_vol_targeting 통합 테스트."""

    def test_scales_weights(self) -> None:
        """가중치가 스케일링됨을 확인."""
        np.random.seed(42)
        returns_a = np.random.normal(0, 0.02, 100).tolist()
        returns_b = np.random.normal(0, 0.01, 100).tolist()

        weights = {"pod-a": 0.5, "pod-b": 0.5}
        pod_returns = {"pod-a": returns_a, "pod-b": returns_b}

        scaled, scalar = apply_vol_targeting(
            weights=weights,
            pod_returns=pod_returns,
            target_vol=0.15,
            lookback=60,
        )

        assert scalar > 0.0
        # 스케일링된 가중치 비율은 유지
        ratio_original = weights["pod-a"] / weights["pod-b"]
        ratio_scaled = scaled["pod-a"] / scaled["pod-b"]
        assert abs(ratio_original - ratio_scaled) < _TOLERANCE

    def test_insufficient_data_returns_cap(self) -> None:
        """데이터 부족 시 cap으로 스케일링."""
        weights = {"pod-a": 0.5, "pod-b": 0.5}
        pod_returns = {"pod-a": [0.01], "pod-b": [0.02]}

        scaled, scalar = apply_vol_targeting(
            weights=weights,
            pod_returns=pod_returns,
            target_vol=0.15,
            lookback=60,
            cap=1.5,
        )

        assert abs(scalar - 1.5) < _TOLERANCE
        assert abs(scaled["pod-a"] - 0.75) < _TOLERANCE

    def test_preserves_zero_weights(self) -> None:
        """weight=0인 pod은 0으로 유지."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 100).tolist()
        weights = {"pod-a": 0.5, "pod-b": 0.0}
        pod_returns = {"pod-a": returns, "pod-b": returns}

        scaled, _ = apply_vol_targeting(
            weights=weights,
            pod_returns=pod_returns,
            target_vol=0.15,
            lookback=60,
        )
        assert scaled["pod-b"] == 0.0
