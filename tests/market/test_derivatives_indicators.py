"""Tests for crypto derivatives indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market.indicators import (
    basis_spread,
    funding_rate_ma,
    funding_zscore,
    liquidation_intensity,
    ls_ratio_zscore,
    oi_momentum,
    oi_price_divergence,
)


@pytest.fixture
def sample_index() -> pd.DatetimeIndex:
    """200일 DatetimeIndex."""
    return pd.date_range("2024-01-01", periods=200, freq="D")


@pytest.fixture
def funding_rate_series(sample_index: pd.DatetimeIndex) -> pd.Series:
    """샘플 펀딩비 시리즈 (8h 기준, ±0.01% 범위)."""
    np.random.seed(42)
    return pd.Series(
        np.random.randn(len(sample_index)) * 0.0001,
        index=sample_index,
    )


@pytest.fixture
def close_series(sample_index: pd.DatetimeIndex) -> pd.Series:
    """샘플 종가 시리즈."""
    np.random.seed(42)
    return pd.Series(
        50000 + np.cumsum(np.random.randn(len(sample_index)) * 500),
        index=sample_index,
    )


@pytest.fixture
def oi_series(sample_index: pd.DatetimeIndex) -> pd.Series:
    """샘플 OI 시리즈."""
    np.random.seed(43)
    return pd.Series(
        1_000_000 + np.cumsum(np.random.randn(len(sample_index)) * 10000),
        index=sample_index,
    )


class TestFundingRateMa:
    """funding_rate_ma 테스트."""

    def test_basic(self, funding_rate_series: pd.Series) -> None:
        """이동평균 계산 확인."""
        result = funding_rate_ma(funding_rate_series, window=3)

        assert len(result) == len(funding_rate_series)
        # window 이전은 NaN
        assert result.iloc[:2].isna().all()
        # window 이후 유효값 존재
        assert result.dropna().shape[0] > 0

    def test_equals_rolling_mean(self, funding_rate_series: pd.Series) -> None:
        """pandas rolling mean과 동일."""
        result = funding_rate_ma(funding_rate_series, window=5)
        expected = funding_rate_series.rolling(5, min_periods=5).mean()
        pd.testing.assert_series_equal(result, expected)


class TestFundingZscore:
    """funding_zscore 테스트."""

    def test_basic(self, funding_rate_series: pd.Series) -> None:
        """z-score 계산 확인."""
        result = funding_zscore(funding_rate_series, ma_window=3, zscore_window=30)

        assert len(result) == len(funding_rate_series)
        valid = result.dropna()
        assert len(valid) > 0

    def test_zscore_range(self, funding_rate_series: pd.Series) -> None:
        """정상 분포에서 z-score 대부분 ±3 범위."""
        result = funding_zscore(funding_rate_series, ma_window=3, zscore_window=60)
        valid = result.dropna()
        # 극단값 허용하되 대부분은 합리적 범위
        assert valid.abs().median() < 3.0


class TestOiMomentum:
    """oi_momentum 테스트."""

    def test_basic(self, oi_series: pd.Series) -> None:
        """OI 변화율 계산."""
        result = oi_momentum(oi_series, period=5)

        assert len(result) == len(oi_series)
        assert result.iloc[:5].isna().all()
        valid = result.dropna()
        assert len(valid) > 0

    def test_known_value(self) -> None:
        """알려진 값 검증."""
        oi = pd.Series([100.0, 110.0, 121.0])
        result = oi_momentum(oi, period=1)

        np.testing.assert_almost_equal(result.iloc[1], 0.1)  # (110-100)/100
        np.testing.assert_almost_equal(result.iloc[2], 0.1)  # (121-110)/110


class TestOiPriceDivergence:
    """oi_price_divergence 테스트."""

    def test_basic(
        self,
        close_series: pd.Series,
        oi_series: pd.Series,
    ) -> None:
        """상관계수 범위 확인 (-1 ~ +1)."""
        result = oi_price_divergence(close_series, oi_series, window=20)

        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= -1.0
        assert valid.max() <= 1.0

    def test_perfect_correlation(self) -> None:
        """완벽 양의 상관 검증."""
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        close = pd.Series(range(50), index=idx, dtype=float)
        oi = pd.Series(range(50), index=idx, dtype=float)

        result = oi_price_divergence(close, oi, window=10)
        valid = result.dropna()
        # 완벽 선형 → 상관 ≈ 1.0
        np.testing.assert_allclose(valid.values, 1.0, atol=1e-10)


class TestBasisSpread:
    """basis_spread 테스트."""

    def test_basic(self) -> None:
        """베이시스 스프레드 계산."""
        spot = pd.Series([50000.0, 51000.0, 49000.0])
        futures = pd.Series([50100.0, 51200.0, 48800.0])

        result = basis_spread(spot, futures)

        # (50100-50000)/50000 * 100 = 0.2%
        np.testing.assert_almost_equal(result.iloc[0], 0.2)
        # (48800-49000)/49000 * 100 ≈ -0.408%
        assert result.iloc[2] < 0  # 백워데이션

    def test_zero_spot(self) -> None:
        """spot=0 → NaN."""
        spot = pd.Series([0.0, 50000.0])
        futures = pd.Series([100.0, 50100.0])

        result = basis_spread(spot, futures)
        assert pd.isna(result.iloc[0])
        assert pd.notna(result.iloc[1])


class TestLsRatioZscore:
    """ls_ratio_zscore 테스트."""

    def test_basic(self, sample_index: pd.DatetimeIndex) -> None:
        """z-score 계산 확인."""
        np.random.seed(44)
        ls_ratio = pd.Series(
            1.0 + np.random.randn(len(sample_index)) * 0.1,
            index=sample_index,
        )

        result = ls_ratio_zscore(ls_ratio, window=30)
        valid = result.dropna()
        assert len(valid) > 0
        # z-score 중앙값 합리적 범위
        assert valid.abs().median() < 3.0

    def test_constant_series(self) -> None:
        """상수 시리즈 → std=0 → NaN."""
        ls = pd.Series([1.5] * 50)
        result = ls_ratio_zscore(ls, window=20)
        # std == 0이면 NaN
        valid = result.dropna()
        assert len(valid) == 0


class TestLiquidationIntensity:
    """liquidation_intensity 테스트."""

    def test_basic(self) -> None:
        """청산 강도 범위 확인."""
        np.random.seed(45)
        n = 100
        volume = pd.Series(np.random.uniform(1000, 5000, n))
        liq_volume = pd.Series(np.random.uniform(0, 500, n))

        result = liquidation_intensity(liq_volume, volume, window=10)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_zero_volume(self) -> None:
        """거래량 0 → NaN."""
        volume = pd.Series([0.0, 1000.0, 0.0, 2000.0])
        liq = pd.Series([10.0, 100.0, 50.0, 200.0])

        result = liquidation_intensity(liq, volume, window=1)
        assert pd.isna(result.iloc[0])  # volume=0
        assert pd.isna(result.iloc[2])  # volume=0
