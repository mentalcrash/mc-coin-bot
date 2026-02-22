"""Tests for options-derived indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market.indicators.options_derived import (
    iv_percentile_rank,
    iv_rv_spread,
)


@pytest.fixture
def sample_index() -> pd.DatetimeIndex:
    """400일 DatetimeIndex."""
    return pd.date_range("2023-01-01", periods=400, freq="D")


class TestIvRvSpread:
    """iv_rv_spread 테스트."""

    def test_basic(self) -> None:
        """기본 스프레드 계산."""
        iv = pd.Series([80.0, 70.0, 90.0])
        rv = pd.Series([60.0, 75.0, 60.0])

        result = iv_rv_spread(iv, rv)
        np.testing.assert_almost_equal(result.iloc[0], 20.0)
        np.testing.assert_almost_equal(result.iloc[1], -5.0)
        np.testing.assert_almost_equal(result.iloc[2], 30.0)

    def test_preserves_nan(self) -> None:
        """NaN 전파."""
        iv = pd.Series([80.0, np.nan, 90.0])
        rv = pd.Series([60.0, 70.0, np.nan])

        result = iv_rv_spread(iv, rv)
        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_empty_series(self) -> None:
        """빈 시리즈."""
        result = iv_rv_spread(pd.Series(dtype=float), pd.Series(dtype=float))
        assert len(result) == 0


class TestIvPercentileRank:
    """iv_percentile_rank 테스트."""

    def test_basic(self, sample_index: pd.DatetimeIndex) -> None:
        """백분위 범위 확인 (0~1)."""
        np.random.seed(42)
        iv = pd.Series(
            np.random.uniform(40, 120, len(sample_index)),
            index=sample_index,
        )
        result = iv_percentile_rank(iv, window=365)

        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_monotonic_increasing(self) -> None:
        """단조 증가 시리즈 → 마지막 값은 1.0."""
        iv = pd.Series(range(1, 366), dtype=float)
        result = iv_percentile_rank(iv, window=365)
        np.testing.assert_almost_equal(result.iloc[-1], 1.0)

    def test_window_too_large(self) -> None:
        """데이터 < 윈도우 → 모두 NaN."""
        iv = pd.Series([50.0, 60.0, 70.0])
        result = iv_percentile_rank(iv, window=365)
        assert result.isna().all()
