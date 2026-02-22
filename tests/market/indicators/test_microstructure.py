"""Tests for microstructure indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market.indicators.microstructure import (
    cvd_price_divergence,
    liquidation_asymmetry,
    liquidation_cascade_score,
    order_flow_imbalance,
    vpin,
)


@pytest.fixture
def sample_index() -> pd.DatetimeIndex:
    """100일 DatetimeIndex."""
    return pd.date_range("2024-01-01", periods=100, freq="D")


class TestVpin:
    """vpin 테스트."""

    def test_basic(self, sample_index: pd.DatetimeIndex) -> None:
        """VPIN 범위 확인 (0~1)."""
        np.random.seed(42)
        close = pd.Series(
            50000 + np.cumsum(np.random.randn(len(sample_index)) * 500),
            index=sample_index,
        )
        volume = pd.Series(
            np.random.uniform(100, 1000, len(sample_index)),
            index=sample_index,
        )
        result = vpin(close, volume, window=50)

        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0 + 1e-10

    def test_all_up(self) -> None:
        """모두 상승 → 모든 거래량이 매수 → VPIN=1."""
        close = pd.Series(range(1, 52), dtype=float)
        volume = pd.Series([100.0] * 51)

        result = vpin(close, volume, window=50)
        # 첫 bar는 diff NaN → 윈도우 내 50개 중 49개 상승 + 1개 NaN
        # 마지막 값은 ≈ 1.0에 가까워야 함
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.iloc[-1] > 0.9

    def test_zero_volume(self) -> None:
        """거래량 0 → NaN."""
        close = pd.Series([100.0, 101.0, 102.0])
        volume = pd.Series([0.0, 0.0, 0.0])
        result = vpin(close, volume, window=3)
        assert result.iloc[-1] == 0.0 or pd.isna(result.iloc[-1])

    def test_empty_series(self) -> None:
        """빈 시리즈."""
        result = vpin(pd.Series(dtype=float), pd.Series(dtype=float))
        assert len(result) == 0


class TestCvdPriceDivergence:
    """cvd_price_divergence 테스트."""

    def test_basic(self) -> None:
        """divergence score 범위 확인."""
        np.random.seed(42)
        n = 30
        close = pd.Series(50000 + np.cumsum(np.random.randn(n) * 500))
        cvd = pd.Series(np.cumsum(np.random.randn(n) * 1000))

        result = cvd_price_divergence(close, cvd, window=14)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_same_direction(self) -> None:
        """같은 방향 → divergence = 0."""
        close = pd.Series(range(1, 20), dtype=float)
        cvd = pd.Series(range(1, 20), dtype=float)

        result = cvd_price_divergence(close, cvd, window=5)
        # 둘 다 상승 → sign 같음 → 차이 0
        valid = result.dropna()
        assert (valid == 0).all()


class TestLiquidationCascadeScore:
    """liquidation_cascade_score 테스트."""

    def test_basic(self) -> None:
        """cascade score 기본 계산."""
        liq_long = pd.Series([100.0, 500.0, 200.0])
        liq_short = pd.Series([50.0, 300.0, 100.0])
        oi = pd.Series([10000.0, 10000.0, 10000.0])

        result = liquidation_cascade_score(liq_long, liq_short, oi)
        np.testing.assert_almost_equal(result.iloc[0], 0.015)
        np.testing.assert_almost_equal(result.iloc[1], 0.08)
        np.testing.assert_almost_equal(result.iloc[2], 0.03)

    def test_zero_oi(self) -> None:
        """OI=0 → NaN."""
        liq_long = pd.Series([100.0])
        liq_short = pd.Series([50.0])
        oi = pd.Series([0.0])
        result = liquidation_cascade_score(liq_long, liq_short, oi)
        assert pd.isna(result.iloc[0])

    def test_zero_liquidations(self) -> None:
        """청산 없음 → 0."""
        liq_long = pd.Series([0.0])
        liq_short = pd.Series([0.0])
        oi = pd.Series([10000.0])
        result = liquidation_cascade_score(liq_long, liq_short, oi)
        np.testing.assert_almost_equal(result.iloc[0], 0.0)


class TestLiquidationAsymmetry:
    """liquidation_asymmetry 테스트."""

    def test_basic(self) -> None:
        """비대칭도 계산."""
        liq_long = pd.Series([100.0, 0.0, 50.0])
        liq_short = pd.Series([100.0, 100.0, 50.0])

        result = liquidation_asymmetry(liq_long, liq_short)
        np.testing.assert_almost_equal(result.iloc[0], 0.0)   # 동일
        np.testing.assert_almost_equal(result.iloc[1], -1.0)  # 숏만
        np.testing.assert_almost_equal(result.iloc[2], 0.0)   # 동일

    def test_range(self) -> None:
        """결과 범위 -1 ~ +1."""
        np.random.seed(42)
        n = 100
        liq_long = pd.Series(np.random.uniform(0, 1000, n))
        liq_short = pd.Series(np.random.uniform(0, 1000, n))

        result = liquidation_asymmetry(liq_long, liq_short)
        valid = result.dropna()
        assert valid.min() >= -1.0 - 1e-10
        assert valid.max() <= 1.0 + 1e-10

    def test_both_zero(self) -> None:
        """둘 다 0 → NaN."""
        liq_long = pd.Series([0.0])
        liq_short = pd.Series([0.0])
        result = liquidation_asymmetry(liq_long, liq_short)
        assert pd.isna(result.iloc[0])


class TestOrderFlowImbalance:
    """order_flow_imbalance 테스트."""

    def test_basic(self) -> None:
        """z-score 계산 확인."""
        np.random.seed(42)
        n = 100
        buy_vol = pd.Series(np.random.uniform(1e6, 5e6, n))
        sell_vol = pd.Series(np.random.uniform(1e6, 5e6, n))

        result = order_flow_imbalance(buy_vol, sell_vol, window=30)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.abs().median() < 3.0

    def test_constant_imbalance(self) -> None:
        """상수 불균형 → std=0 → NaN."""
        buy_vol = pd.Series([1e6] * 50)
        sell_vol = pd.Series([5e5] * 50)
        result = order_flow_imbalance(buy_vol, sell_vol, window=30)
        valid = result.dropna()
        assert len(valid) == 0

    def test_empty_series(self) -> None:
        """빈 시리즈."""
        result = order_flow_imbalance(
            pd.Series(dtype=float), pd.Series(dtype=float)
        )
        assert len(result) == 0
