"""Tests for macro-derived indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market.indicators.macro_derived import (
    btc_spy_correlation,
    credit_spread_proxy,
    risk_appetite_index,
)


@pytest.fixture
def sample_index() -> pd.DatetimeIndex:
    """200일 DatetimeIndex."""
    return pd.date_range("2024-01-01", periods=200, freq="D")


@pytest.fixture
def price_series(sample_index: pd.DatetimeIndex) -> pd.Series:
    """랜덤워크 가격 시리즈."""
    np.random.seed(42)
    return pd.Series(
        100 + np.cumsum(np.random.randn(len(sample_index)) * 0.5),
        index=sample_index,
    )


class TestCreditSpreadProxy:
    """credit_spread_proxy 테스트."""

    def test_basic(self, sample_index: pd.DatetimeIndex) -> None:
        """기본 계산 확인."""
        np.random.seed(42)
        hyg = pd.Series(
            80 + np.cumsum(np.random.randn(len(sample_index)) * 0.3),
            index=sample_index,
        )
        tlt = pd.Series(
            100 + np.cumsum(np.random.randn(len(sample_index)) * 0.5),
            index=sample_index,
        )
        result = credit_spread_proxy(hyg, tlt, window=20)

        assert len(result) == len(sample_index)
        # pct_change 첫 행 NaN + window-1 NaN
        valid = result.dropna()
        assert len(valid) > 0

    def test_identical_series(self, sample_index: pd.DatetimeIndex) -> None:
        """동일 시리즈 → spread ≈ 0."""
        price = pd.Series(
            100 + np.arange(len(sample_index), dtype=float),
            index=sample_index,
        )
        result = credit_spread_proxy(price, price, window=20)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-15)


class TestRiskAppetiteIndex:
    """risk_appetite_index 테스트."""

    def test_basic(self, sample_index: pd.DatetimeIndex) -> None:
        """z-score 계산 확인."""
        np.random.seed(42)
        spy = pd.Series(
            450 + np.cumsum(np.random.randn(len(sample_index)) * 2),
            index=sample_index,
        )
        tlt = pd.Series(
            100 + np.cumsum(np.random.randn(len(sample_index)) * 1),
            index=sample_index,
        )
        result = risk_appetite_index(spy, tlt, window=60)

        valid = result.dropna()
        assert len(valid) > 0
        # z-score 중앙값 합리적 범위
        assert valid.abs().median() < 3.0

    def test_constant_diff(self) -> None:
        """상수 차이 → std=0 → NaN."""
        spy = pd.Series([100.0] * 100)
        tlt = pd.Series([50.0] * 100)
        result = risk_appetite_index(spy, tlt, window=60)
        # pct_change 후 0, 0, ... → std=0 → NaN
        valid = result.dropna()
        assert len(valid) == 0


class TestBtcSpyCorrelation:
    """btc_spy_correlation 테스트."""

    def test_basic(self, sample_index: pd.DatetimeIndex) -> None:
        """상관 범위 확인 (-1 ~ +1)."""
        np.random.seed(42)
        btc = pd.Series(
            50000 + np.cumsum(np.random.randn(len(sample_index)) * 500),
            index=sample_index,
        )
        spy = pd.Series(
            450 + np.cumsum(np.random.randn(len(sample_index)) * 2),
            index=sample_index,
        )
        result = btc_spy_correlation(btc, spy, window=21)

        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= -1.0 - 1e-10
        assert valid.max() <= 1.0 + 1e-10

    def test_perfect_correlation(self) -> None:
        """완벽 양의 상관."""
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        btc = pd.Series(range(50), index=idx, dtype=float)
        spy = pd.Series(range(50), index=idx, dtype=float)

        result = btc_spy_correlation(btc, spy, window=10)
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-10)

    def test_empty_series(self) -> None:
        """빈 시리즈."""
        result = btc_spy_correlation(
            pd.Series(dtype=float),
            pd.Series(dtype=float),
            window=21,
        )
        assert len(result) == 0
