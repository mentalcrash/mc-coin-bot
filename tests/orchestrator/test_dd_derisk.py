"""Tests for Drawdown-Based De-Risking Overlay."""

from __future__ import annotations

import pytest

from src.orchestrator.dd_derisk import apply_dd_derisk

# ── Constants ────────────────────────────────────────────────────

_HALF = 0.10
_ZERO = 0.20


# ── TestApplyDdDerisk ────────────────────────────────────────────


class TestApplyDdDerisk:
    def test_normal_no_change(self) -> None:
        """DD < half → 변경 없음."""
        weights = {"pod-a": 0.4, "pod-b": 0.6}
        dds = {"pod-a": 0.05, "pod-b": 0.03}
        adj, actions = apply_dd_derisk(weights, dds, _HALF, _ZERO)
        assert adj == weights
        assert actions == {"pod-a": "normal", "pod-b": "normal"}

    def test_half_threshold(self) -> None:
        """DD = 0.12 (> 0.10) → weight * 0.5."""
        weights = {"pod-a": 0.4}
        dds = {"pod-a": 0.12}
        adj, actions = apply_dd_derisk(weights, dds, _HALF, _ZERO)
        assert adj["pod-a"] == pytest.approx(0.2)
        assert actions["pod-a"] == "halved"

    def test_zero_threshold(self) -> None:
        """DD = 0.22 (> 0.20) → weight = 0.0."""
        weights = {"pod-a": 0.4}
        dds = {"pod-a": 0.22}
        adj, actions = apply_dd_derisk(weights, dds, _HALF, _ZERO)
        assert adj["pod-a"] == 0.0
        assert actions["pod-a"] == "zeroed"

    def test_mixed_pods(self) -> None:
        """Pod별 다른 DD → 각각 적용."""
        weights = {"pod-a": 0.3, "pod-b": 0.3, "pod-c": 0.4}
        dds = {"pod-a": 0.05, "pod-b": 0.15, "pod-c": 0.25}
        adj, actions = apply_dd_derisk(weights, dds, _HALF, _ZERO)
        assert adj["pod-a"] == pytest.approx(0.3)
        assert adj["pod-b"] == pytest.approx(0.15)
        assert adj["pod-c"] == 0.0
        assert actions == {"pod-a": "normal", "pod-b": "halved", "pod-c": "zeroed"}

    def test_boundary_at_half(self) -> None:
        """DD = 0.10 (정확히 경계) → half 적용."""
        weights = {"pod-a": 0.5}
        dds = {"pod-a": 0.10}
        adj, actions = apply_dd_derisk(weights, dds, _HALF, _ZERO)
        assert adj["pod-a"] == pytest.approx(0.25)
        assert actions["pod-a"] == "halved"

    def test_boundary_at_zero(self) -> None:
        """DD = 0.20 (정확히 경계) → zero 적용."""
        weights = {"pod-a": 0.5}
        dds = {"pod-a": 0.20}
        adj, actions = apply_dd_derisk(weights, dds, _HALF, _ZERO)
        assert adj["pod-a"] == 0.0
        assert actions["pod-a"] == "zeroed"

    def test_empty_weights(self) -> None:
        """빈 input → 빈 output."""
        adj, actions = apply_dd_derisk({}, {}, _HALF, _ZERO)
        assert adj == {}
        assert actions == {}

    def test_missing_drawdown(self) -> None:
        """drawdowns에 없는 pod → 변경 없음."""
        weights = {"pod-a": 0.5, "pod-b": 0.5}
        dds = {"pod-a": 0.05}  # pod-b 없음
        adj, actions = apply_dd_derisk(weights, dds, _HALF, _ZERO)
        assert adj["pod-b"] == pytest.approx(0.5)
        assert actions["pod-b"] == "normal"
