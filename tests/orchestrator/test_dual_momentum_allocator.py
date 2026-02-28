"""Tests for Dual Momentum allocation in IntraPodAllocator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.orchestrator.asset_allocator import (
    AssetAllocationConfig,
    IntraPodAllocator,
)
from src.orchestrator.models import AssetAllocationMethod

# ── Helpers ─────────────────────────────────────────────────────


def _make_dm_allocator(
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"),
    **overrides: Any,
) -> IntraPodAllocator:
    defaults: dict[str, Any] = {
        "method": AssetAllocationMethod.DUAL_MOMENTUM,
        "rebalance_bars": 1,
        "mom_lookback": 10,
        "top_pct": 0.50,
        "abs_mom_threshold": 0.0,
        "exposure_floor": 0.0,
    }
    defaults.update(overrides)
    config = AssetAllocationConfig(**defaults)
    return IntraPodAllocator(config=config, symbols=symbols)


def _make_trending_returns(
    symbols: tuple[str, ...],
    n: int = 20,
    drifts: list[float] | None = None,
    seed: int = 42,
) -> dict[str, list[float]]:
    """심볼별 다른 drift를 가진 수익률 생성."""
    rng = np.random.default_rng(seed)
    if drifts is None:
        drifts = [0.01 * (i + 1) for i in range(len(symbols))]
    result: dict[str, list[float]] = {}
    for i, s in enumerate(symbols):
        result[s] = (rng.normal(drifts[i], 0.005, n)).tolist()
    return result


# ── TestCrossSectionalRanking ────────────────────────────────────


class TestCrossSectionalRanking:
    """Cross-sectional ranking 정확성 테스트."""

    def test_top_half_selected(self) -> None:
        """상위 50% 심볼만 비중 배분."""
        symbols = ("A", "B", "C", "D")
        alloc = _make_dm_allocator(symbols=symbols, top_pct=0.50)
        # A: 강한 양수, B: 약한 양수, C: 약한 음수, D: 강한 음수
        returns = _make_trending_returns(symbols, n=10, drifts=[0.05, 0.02, -0.01, -0.03])
        weights = alloc.on_bar(returns)

        # A, B 선택 (top 50%)
        assert weights["A"] > 0
        assert weights["B"] > 0
        assert weights["C"] == 0.0
        assert weights["D"] == 0.0

    def test_bottom_symbols_zero_weight(self) -> None:
        """하위 심볼은 weight=0 보장."""
        symbols = ("X", "Y", "Z")
        alloc = _make_dm_allocator(symbols=symbols, top_pct=0.34)
        # top_pct=0.34 → n_long = max(1, round(3*0.34)) = 1
        returns = _make_trending_returns(symbols, n=10, drifts=[0.05, 0.01, -0.02])
        weights = alloc.on_bar(returns)

        assert weights["X"] > 0  # 최상위만 선택
        assert weights["Y"] == 0.0
        assert weights["Z"] == 0.0

    def test_selected_symbols_equal_weight(self) -> None:
        """선택된 심볼 간 EW 배분."""
        symbols = ("A", "B", "C", "D")
        alloc = _make_dm_allocator(symbols=symbols, top_pct=0.50)
        returns = _make_trending_returns(symbols, n=10, drifts=[0.05, 0.02, -0.01, -0.03])
        weights = alloc.on_bar(returns)

        # top 2 symbols should each get 0.5
        selected = [w for w in weights.values() if w > 0]
        assert len(selected) == 2
        for w in selected:
            assert pytest.approx(0.5) == w

    def test_ranking_order_respected(self) -> None:
        """momentum 순서대로 ranking."""
        symbols = ("A", "B", "C", "D")
        alloc = _make_dm_allocator(symbols=symbols, top_pct=0.25)
        # A가 가장 강한 drift
        returns = _make_trending_returns(symbols, n=10, drifts=[0.10, 0.03, 0.01, -0.05])
        weights = alloc.on_bar(returns)

        # n_long = max(1, round(4*0.25)) = 1
        assert weights["A"] == pytest.approx(1.0)
        assert weights["B"] == 0.0
        assert weights["C"] == 0.0
        assert weights["D"] == 0.0


# ── TestAbsoluteMomentumGate ─────────────────────────────────────


class TestAbsoluteMomentumGate:
    """절대 모멘텀 게이트 테스트."""

    def test_all_positive_full_exposure(self) -> None:
        """전체 양수 momentum → exposure=1.0."""
        symbols = ("A", "B")
        alloc = _make_dm_allocator(symbols=symbols, abs_mom_threshold=0.0)
        returns = _make_trending_returns(symbols, n=10, drifts=[0.05, 0.03])
        alloc.on_bar(returns)
        assert alloc.exposure == pytest.approx(1.0)

    def test_all_negative_zero_exposure(self) -> None:
        """전체 음수 momentum → exposure=floor."""
        symbols = ("A", "B")
        alloc = _make_dm_allocator(symbols=symbols, abs_mom_threshold=0.0, exposure_floor=0.0)
        returns = _make_trending_returns(symbols, n=10, drifts=[-0.05, -0.03])
        alloc.on_bar(returns)
        assert alloc.exposure == pytest.approx(0.0)

    def test_negative_with_floor(self) -> None:
        """음수 momentum + floor=0.2 → exposure=0.2."""
        symbols = ("A", "B")
        alloc = _make_dm_allocator(symbols=symbols, abs_mom_threshold=0.0, exposure_floor=0.2)
        returns = _make_trending_returns(symbols, n=10, drifts=[-0.05, -0.03])
        alloc.on_bar(returns)
        assert alloc.exposure == pytest.approx(0.2)

    def test_exposure_floor_respected(self) -> None:
        """exposure는 항상 floor 이상."""
        symbols = ("A", "B", "C")
        floor = 0.15
        alloc = _make_dm_allocator(symbols=symbols, abs_mom_threshold=0.0, exposure_floor=floor)
        returns = _make_trending_returns(symbols, n=10, drifts=[-0.10, -0.08, -0.05])
        alloc.on_bar(returns)
        assert alloc.exposure >= floor


# ── TestDataInsufficiency ────────────────────────────────────────


class TestDataInsufficiency:
    """데이터 부족 시 fallback 테스트."""

    def test_insufficient_data_returns_ew(self) -> None:
        """lookback보다 짧은 데이터 → EW fallback."""
        symbols = ("A", "B", "C")
        alloc = _make_dm_allocator(symbols=symbols, mom_lookback=20)
        # Only 5 data points (< 20)
        returns = {s: [0.01] * 5 for s in symbols}
        weights = alloc.on_bar(returns)

        for w in weights.values():
            assert pytest.approx(1.0 / 3) == w

    def test_insufficient_data_exposure_one(self) -> None:
        """데이터 부족 시 exposure=1.0."""
        symbols = ("A", "B")
        alloc = _make_dm_allocator(symbols=symbols, mom_lookback=20)
        returns = {s: [0.01] * 5 for s in symbols}
        alloc.on_bar(returns)
        assert alloc.exposure == pytest.approx(1.0)

    def test_empty_returns_fallback(self) -> None:
        """빈 returns → EW fallback."""
        symbols = ("A", "B")
        alloc = _make_dm_allocator(symbols=symbols, mom_lookback=10)
        returns: dict[str, list[float]] = {s: [] for s in symbols}
        weights = alloc.on_bar(returns)
        assert pytest.approx(0.5) == weights["A"]


# ── TestEdgeCases ────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case 테스트."""

    def test_all_negative_momentum(self) -> None:
        """전체 음수 momentum → 여전히 top_pct 선택 (상대 ranking)."""
        symbols = ("A", "B", "C", "D")
        alloc = _make_dm_allocator(symbols=symbols, top_pct=0.50)
        returns = _make_trending_returns(symbols, n=10, drifts=[-0.01, -0.02, -0.05, -0.08])
        weights = alloc.on_bar(returns)

        # 상대적으로 덜 나쁜 A, B가 선택됨
        selected = [s for s, w in weights.items() if w > 0]
        assert len(selected) == 2
        assert "A" in selected
        assert "B" in selected

    def test_equal_momentum(self) -> None:
        """동일 momentum → 안정적으로 동작."""
        symbols = ("A", "B", "C", "D")
        alloc = _make_dm_allocator(symbols=symbols, top_pct=0.50)
        # 모두 동일한 수익률
        shared = [0.01] * 10
        returns = {s: shared.copy() for s in symbols}
        weights = alloc.on_bar(returns)

        # 모두 동일 → top 2 선택 (정렬 안정성에 의존)
        selected = [s for s, w in weights.items() if w > 0]
        assert len(selected) == 2
        assert pytest.approx(1.0) == sum(weights.values())

    def test_two_symbols(self) -> None:
        """2심볼 + top_pct=0.5 → 1개 선택."""
        symbols = ("A", "B")
        alloc = _make_dm_allocator(symbols=symbols, top_pct=0.50)
        returns = _make_trending_returns(symbols, n=10, drifts=[0.05, -0.02])
        weights = alloc.on_bar(returns)

        # n_long = max(1, round(2*0.5)) = 1
        assert weights["A"] == pytest.approx(1.0)
        assert weights["B"] == 0.0

    def test_eight_symbols(self) -> None:
        """8심볼 + top_pct=0.25 → 2개 선택."""
        symbols = tuple(f"S{i}" for i in range(8))
        drifts = [0.08, 0.06, 0.04, 0.02, -0.01, -0.02, -0.04, -0.06]
        alloc = _make_dm_allocator(symbols=symbols, top_pct=0.25)
        returns = _make_trending_returns(symbols, n=10, drifts=drifts)
        weights = alloc.on_bar(returns)

        # n_long = max(1, round(8*0.25)) = 2
        selected = [s for s, w in weights.items() if w > 0]
        assert len(selected) == 2
        assert pytest.approx(1.0) == sum(weights.values())

    def test_single_symbol(self) -> None:
        """1심볼 → 항상 weight=1.0."""
        symbols = ("A",)
        alloc = _make_dm_allocator(symbols=symbols, top_pct=0.50)
        returns = {"A": [0.01] * 10}
        weights = alloc.on_bar(returns)
        assert weights["A"] == pytest.approx(1.0)


# ── TestSerialization ────────────────────────────────────────────


class TestSerialization:
    """Serialization roundtrip 테스트."""

    def test_exposure_serialized(self) -> None:
        """to_dict에 exposure 포함."""
        symbols = ("A", "B")
        alloc = _make_dm_allocator(symbols=symbols, abs_mom_threshold=0.0, exposure_floor=0.0)
        returns = _make_trending_returns(symbols, n=10, drifts=[-0.05, -0.03])
        alloc.on_bar(returns)

        data = alloc.to_dict()
        assert "exposure" in data
        assert data["exposure"] == pytest.approx(0.0)

    def test_roundtrip_preserves_state(self) -> None:
        """to_dict → restore_from_dict 상태 보존."""
        symbols = ("A", "B", "C")
        alloc = _make_dm_allocator(symbols=symbols, exposure_floor=0.1)
        returns = _make_trending_returns(symbols, n=10, drifts=[-0.05, 0.01, 0.03])
        alloc.on_bar(returns)

        original_weights = alloc.weights
        original_exposure = alloc.exposure
        original_bar_count = alloc.bar_count

        data = alloc.to_dict()

        alloc2 = _make_dm_allocator(symbols=symbols, exposure_floor=0.1)
        alloc2.restore_from_dict(data)

        assert alloc2.weights == original_weights
        assert alloc2.exposure == pytest.approx(original_exposure)
        assert alloc2.bar_count == original_bar_count


# ── TestExposureForOtherMethods ──────────────────────────────────


class TestExposureForOtherMethods:
    """기존 method는 exposure=1.0 유지."""

    @pytest.mark.parametrize(
        "method",
        [
            AssetAllocationMethod.EQUAL_WEIGHT,
            AssetAllocationMethod.INVERSE_VOLATILITY,
            AssetAllocationMethod.SIGNAL_WEIGHTED,
        ],
    )
    def test_exposure_always_one(self, method: AssetAllocationMethod) -> None:
        symbols = ("A", "B")
        config = AssetAllocationConfig(
            method=method,
            rebalance_bars=1,
            vol_lookback=10,
        )
        alloc = IntraPodAllocator(config=config, symbols=symbols)
        returns = {"A": [0.01, -0.02, 0.015] * 10, "B": [-0.01, 0.02, -0.005] * 10}
        strengths = {"A": 0.5, "B": 0.5}
        alloc.on_bar(returns, strengths=strengths)
        assert alloc.exposure == pytest.approx(1.0)
