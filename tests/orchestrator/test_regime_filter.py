"""Tests for Regime-Adaptive Cash Buffer."""

from __future__ import annotations

import numpy as np
import pytest

from src.orchestrator.regime_filter import (
    apply_cash_buffer,
    compute_adx,
    compute_regime_cash_buffer,
)

# ── Constants ─────────────────────────────────────────────────────

_TOLERANCE = 1e-6


# ── compute_adx ──────────────────────────────────────────────────


class TestComputeAdx:
    """compute_adx 단위 테스트."""

    def test_trending_market_high_adx(self) -> None:
        """강한 상승 추세 → 높은 ADX."""
        n = 100
        # 꾸준히 상승하는 시장
        base = np.cumsum(np.random.RandomState(42).uniform(0.5, 1.5, n)) + 100
        high = (base + np.random.RandomState(42).uniform(0.5, 2.0, n)).tolist()
        low = (base - np.random.RandomState(42).uniform(0.5, 2.0, n)).tolist()
        close = base.tolist()

        adx = compute_adx(high, low, close, period=14)
        assert adx > 20.0  # 추세 시장

    def test_range_bound_market_low_adx(self) -> None:
        """횡보 시장 → 낮은 ADX."""
        n = 100
        rng = np.random.RandomState(123)
        # 횡보: sin 파동 + 작은 노이즈
        base = 100 + 2 * np.sin(np.linspace(0, 8 * np.pi, n)) + rng.normal(0, 0.1, n)
        high = (base + rng.uniform(0.3, 0.8, n)).tolist()
        low = (base - rng.uniform(0.3, 0.8, n)).tolist()
        close = base.tolist()

        adx = compute_adx(high, low, close, period=14)
        assert adx < 30.0  # 횡보 시장

    def test_insufficient_data(self) -> None:
        """데이터 부족 → 0.0."""
        adx = compute_adx([100, 101], [99, 100], [100, 101], period=14)
        assert adx == 0.0

    def test_zero_period(self) -> None:
        """period가 작아도 안전하게 처리."""
        n = 50
        high = [float(100 + i * 0.5) for i in range(n)]
        low = [float(99 + i * 0.5) for i in range(n)]
        close = [float(99.5 + i * 0.5) for i in range(n)]

        adx = compute_adx(high, low, close, period=5)
        assert adx >= 0.0

    def test_adx_range(self) -> None:
        """ADX는 0~100 범위."""
        n = 200
        rng = np.random.RandomState(42)
        base = np.cumsum(rng.normal(0, 1, n)) + 100
        high = (base + rng.uniform(0.5, 2, n)).tolist()
        low = (base - rng.uniform(0.5, 2, n)).tolist()
        close = base.tolist()

        adx = compute_adx(high, low, close, period=14)
        assert 0.0 <= adx <= 100.0


# ── compute_regime_cash_buffer ────────────────────────────────────


class TestComputeRegimeCashBuffer:
    """compute_regime_cash_buffer 단위 테스트."""

    def _make_trending_data(self, n: int = 100) -> tuple[list[float], list[float], list[float]]:
        """강한 상승 추세 데이터 생성."""
        rng = np.random.RandomState(42)
        base = np.cumsum(rng.uniform(0.5, 1.5, n)) + 100
        high = (base + rng.uniform(0.5, 2.0, n)).tolist()
        low = (base - rng.uniform(0.5, 2.0, n)).tolist()
        close = base.tolist()
        return high, low, close

    def _make_ranging_data(self, n: int = 100) -> tuple[list[float], list[float], list[float]]:
        """횡보 데이터 생성."""
        rng = np.random.RandomState(123)
        base = 100 + 2 * np.sin(np.linspace(0, 8 * np.pi, n)) + rng.normal(0, 0.1, n)
        high = (base + rng.uniform(0.3, 0.8, n)).tolist()
        low = (base - rng.uniform(0.3, 0.8, n)).tolist()
        close = base.tolist()
        return high, low, close

    def test_trending_market_no_buffer(self) -> None:
        """강한 추세 → cash buffer ≈ 0."""
        high, low, close = self._make_trending_data()
        buffer, avg_adx = compute_regime_cash_buffer(
            price_histories={"BTC": close},
            high_histories={"BTC": high},
            low_histories={"BTC": low},
            adx_period=14,
            trend_threshold=25.0,
            range_threshold=20.0,
            max_cash_buffer=0.40,
        )
        # 추세 시장이면 buffer가 낮아야 함
        assert buffer < 0.20

    def test_empty_data(self) -> None:
        """빈 데이터 → (0.0, 0.0)."""
        buffer, avg_adx = compute_regime_cash_buffer(
            price_histories={},
            high_histories={},
            low_histories={},
        )
        assert buffer == 0.0
        assert avg_adx == 0.0

    def test_linear_interpolation(self) -> None:
        """ADX가 range~trend 사이 → 선형 보간."""
        high, low, close = self._make_ranging_data()
        # 횡보 데이터의 ADX 기반으로 threshold를 설정
        adx_val = compute_adx(high, low, close, period=14)

        # threshold를 ADX보다 크게 설정하여 buffer 발생 보장
        buffer, _ = compute_regime_cash_buffer(
            price_histories={"BTC": close},
            high_histories={"BTC": high},
            low_histories={"BTC": low},
            adx_period=14,
            trend_threshold=adx_val + 10.0,
            range_threshold=adx_val - 10.0 if adx_val > 10.0 else 0.0,
            max_cash_buffer=0.40,
        )
        # ADX가 range~trend 사이이므로 0 < buffer < max
        assert 0.0 < buffer < 0.40

    def test_multi_symbol_average(self) -> None:
        """다중 심볼 평균 ADX."""
        high1, low1, close1 = self._make_trending_data()
        high2, low2, close2 = self._make_ranging_data()

        buffer, avg_adx = compute_regime_cash_buffer(
            price_histories={"BTC": close1, "ETH": close2},
            high_histories={"BTC": high1, "ETH": high2},
            low_histories={"BTC": low1, "ETH": low2},
        )
        assert avg_adx > 0.0


# ── apply_cash_buffer ─────────────────────────────────────────────


class TestApplyCashBuffer:
    """apply_cash_buffer 단위 테스트."""

    def test_zero_buffer(self) -> None:
        """buffer=0 → 가중치 변화 없음."""
        weights = {"pod-a": 0.5, "pod-b": 0.5}
        result = apply_cash_buffer(weights, 0.0)
        assert abs(result["pod-a"] - 0.5) < _TOLERANCE
        assert abs(result["pod-b"] - 0.5) < _TOLERANCE

    def test_max_buffer(self) -> None:
        """buffer=0.4 → 60% 투자."""
        weights = {"pod-a": 0.5, "pod-b": 0.5}
        result = apply_cash_buffer(weights, 0.40)
        assert abs(result["pod-a"] - 0.30) < _TOLERANCE
        assert abs(result["pod-b"] - 0.30) < _TOLERANCE

    def test_preserves_ratios(self) -> None:
        """비율 유지 확인."""
        weights = {"pod-a": 0.6, "pod-b": 0.3, "pod-c": 0.1}
        result = apply_cash_buffer(weights, 0.20)
        # 비율 6:3:1 유지
        assert abs(result["pod-a"] / result["pod-b"] - 2.0) < _TOLERANCE
        assert abs(result["pod-b"] / result["pod-c"] - 3.0) < _TOLERANCE

    def test_full_buffer(self) -> None:
        """buffer=1.0 → 모든 가중치 0."""
        weights = {"pod-a": 0.5, "pod-b": 0.5}
        result = apply_cash_buffer(weights, 1.0)
        assert abs(result["pod-a"]) < _TOLERANCE
        assert abs(result["pod-b"]) < _TOLERANCE
