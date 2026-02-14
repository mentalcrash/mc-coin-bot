"""Tests for orchestrator netting — 포지션 넷팅 순수 함수."""

from __future__ import annotations

import pytest

from src.orchestrator.netting import (
    attribute_fill,
    compute_deltas,
    compute_gross_leverage,
    compute_net_weights,
    scale_weights_to_leverage,
)

# ── TestComputeNetWeights ──────────────────────────────────────


class TestComputeNetWeights:
    def test_single_pod_passthrough(self) -> None:
        result = compute_net_weights({"pod-a": {"BTC/USDT": 0.5, "ETH/USDT": 0.3}})
        assert result == {"BTC/USDT": 0.5, "ETH/USDT": 0.3}

    def test_two_pods_same_direction(self) -> None:
        result = compute_net_weights({
            "pod-a": {"BTC/USDT": 0.3},
            "pod-b": {"BTC/USDT": 0.2},
        })
        assert result["BTC/USDT"] == pytest.approx(0.5)

    def test_two_pods_opposite_direction(self) -> None:
        result = compute_net_weights({
            "pod-a": {"BTC/USDT": 0.3},
            "pod-b": {"BTC/USDT": -0.3},
        })
        assert result["BTC/USDT"] == pytest.approx(0.0)

    def test_multiple_symbols(self) -> None:
        result = compute_net_weights({
            "pod-a": {"BTC/USDT": 0.3, "ETH/USDT": -0.2},
            "pod-b": {"BTC/USDT": -0.1, "SOL/USDT": 0.4},
        })
        assert result["BTC/USDT"] == pytest.approx(0.2)
        assert result["ETH/USDT"] == pytest.approx(-0.2)
        assert result["SOL/USDT"] == pytest.approx(0.4)

    def test_empty_input(self) -> None:
        assert compute_net_weights({}) == {}

    def test_empty_pod_weights(self) -> None:
        result = compute_net_weights({"pod-a": {}})
        assert result == {}


# ── TestComputeDeltas ──────────────────────────────────────────


class TestComputeDeltas:
    def test_increase_weight(self) -> None:
        result = compute_deltas({"BTC/USDT": 0.5}, {"BTC/USDT": 0.3})
        assert result["BTC/USDT"] == pytest.approx(0.2)

    def test_decrease_weight(self) -> None:
        result = compute_deltas({"BTC/USDT": 0.2}, {"BTC/USDT": 0.5})
        assert result["BTC/USDT"] == pytest.approx(-0.3)

    def test_new_symbol(self) -> None:
        result = compute_deltas({"ETH/USDT": 0.3}, {})
        assert result["ETH/USDT"] == pytest.approx(0.3)

    def test_close_position(self) -> None:
        result = compute_deltas({}, {"BTC/USDT": 0.5})
        assert result["BTC/USDT"] == pytest.approx(-0.5)

    def test_no_change(self) -> None:
        result = compute_deltas({"BTC/USDT": 0.5}, {"BTC/USDT": 0.5})
        assert result["BTC/USDT"] == pytest.approx(0.0)


# ── TestAttributeFill ──────────────────────────────────────────


class TestAttributeFill:
    def test_single_pod_100pct(self) -> None:
        result = attribute_fill("BTC/USDT", 1.0, 50000.0, 10.0, {"pod-a": 0.5})
        assert result["pod-a"] == pytest.approx((1.0, 50000.0, 10.0))

    def test_two_pods_proportional(self) -> None:
        result = attribute_fill("BTC/USDT", 1.0, 50000.0, 10.0, {"pod-a": 0.6, "pod-b": 0.4})
        assert result["pod-a"][0] == pytest.approx(0.6)
        assert result["pod-b"][0] == pytest.approx(0.4)
        assert result["pod-a"][2] == pytest.approx(6.0)
        assert result["pod-b"][2] == pytest.approx(4.0)

    def test_opposite_direction_by_abs(self) -> None:
        """반대 방향 Pod: |target| 비율로 배분."""
        result = attribute_fill("BTC/USDT", 1.0, 50000.0, 10.0, {"pod-a": 0.3, "pod-b": -0.1})
        # abs(0.3)/(0.3+0.1)=0.75, abs(-0.1)/(0.3+0.1)=0.25
        assert result["pod-a"][0] == pytest.approx(0.75)
        assert result["pod-b"][0] == pytest.approx(0.25)

    def test_empty_targets(self) -> None:
        result = attribute_fill("BTC/USDT", 1.0, 50000.0, 10.0, {})
        assert result == {}

    def test_zero_targets(self) -> None:
        result = attribute_fill("BTC/USDT", 1.0, 50000.0, 10.0, {"pod-a": 0.0})
        assert result == {}

    def test_price_passthrough(self) -> None:
        result = attribute_fill("BTC/USDT", 2.0, 42000.0, 8.0, {"pod-a": 0.5})
        assert result["pod-a"][1] == 42000.0


# ── TestComputeGrossLeverage ───────────────────────────────────


class TestComputeGrossLeverage:
    def test_long_only(self) -> None:
        assert compute_gross_leverage({"BTC/USDT": 1.0, "ETH/USDT": 0.5}) == pytest.approx(1.5)

    def test_mixed_direction(self) -> None:
        assert compute_gross_leverage({"BTC/USDT": 1.0, "ETH/USDT": -0.5}) == pytest.approx(1.5)

    def test_empty(self) -> None:
        assert compute_gross_leverage({}) == pytest.approx(0.0)

    def test_zero_weights(self) -> None:
        assert compute_gross_leverage({"BTC/USDT": 0.0}) == pytest.approx(0.0)


# ── TestScaleWeightsToLeverage ─────────────────────────────────


class TestScaleWeightsToLeverage:
    def test_within_limit_unchanged(self) -> None:
        weights = {"BTC/USDT": 1.0, "ETH/USDT": 0.5}
        result = scale_weights_to_leverage(weights, 3.0)
        assert result == weights

    def test_exceeds_limit_scaled_down(self) -> None:
        weights = {"BTC/USDT": 2.0, "ETH/USDT": 2.0}
        result = scale_weights_to_leverage(weights, 3.0)
        # gross=4.0, scale=3.0/4.0=0.75
        assert result["BTC/USDT"] == pytest.approx(1.5)
        assert result["ETH/USDT"] == pytest.approx(1.5)
        assert compute_gross_leverage(result) == pytest.approx(3.0)

    def test_mixed_direction_scaled(self) -> None:
        weights = {"BTC/USDT": 2.0, "ETH/USDT": -2.0}
        result = scale_weights_to_leverage(weights, 3.0)
        assert compute_gross_leverage(result) == pytest.approx(3.0)
        # 방향 유지
        assert result["BTC/USDT"] > 0
        assert result["ETH/USDT"] < 0

    def test_empty_weights(self) -> None:
        result = scale_weights_to_leverage({}, 3.0)
        assert result == {}

    def test_original_not_mutated(self) -> None:
        original = {"BTC/USDT": 5.0}
        _ = scale_weights_to_leverage(original, 3.0)
        assert original["BTC/USDT"] == 5.0
