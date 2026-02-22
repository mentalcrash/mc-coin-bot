"""Tests for on-chain indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.market.indicators.onchain import (
    exchange_flow_net_zscore,
    mvrv_zscore,
    nvt_signal,
    puell_multiple,
    stablecoin_supply_ratio,
    tvl_stablecoin_ratio,
)


@pytest.fixture
def sample_index() -> pd.DatetimeIndex:
    """400일 DatetimeIndex (365일 윈도우 테스트 가능)."""
    return pd.date_range("2023-01-01", periods=400, freq="D")


class TestPuellMultiple:
    """puell_multiple 테스트."""

    def test_basic(self, sample_index: pd.DatetimeIndex) -> None:
        """기본 계산 확인."""
        np.random.seed(42)
        revenue = pd.Series(
            np.random.uniform(10_000, 50_000, len(sample_index)),
            index=sample_index,
        )
        result = puell_multiple(revenue, ma_window=365)

        assert len(result) == len(revenue)
        # 365일 이전은 NaN
        assert result.iloc[:364].isna().all()
        valid = result.dropna()
        assert len(valid) > 0
        # Puell Multiple은 양수
        assert (valid > 0).all()

    def test_known_value(self) -> None:
        """알려진 값 검증."""
        # 365일 동일 수익 → Puell = 1.0
        revenue = pd.Series([100.0] * 365)
        result = puell_multiple(revenue, ma_window=365)
        np.testing.assert_almost_equal(result.iloc[-1], 1.0)

    def test_zero_ma(self) -> None:
        """MA=0 → NaN."""
        revenue = pd.Series([0.0] * 365)
        result = puell_multiple(revenue, ma_window=365)
        assert pd.isna(result.iloc[-1])

    def test_empty_series(self) -> None:
        """빈 시리즈."""
        result = puell_multiple(pd.Series(dtype=float))
        assert len(result) == 0


class TestNvtSignal:
    """nvt_signal 테스트."""

    def test_basic(self) -> None:
        """기본 계산 확인."""
        np.random.seed(42)
        n = 100
        market_cap = pd.Series(np.random.uniform(1e12, 2e12, n))
        tx_vol = pd.Series(np.random.uniform(1e9, 5e9, n))

        result = nvt_signal(market_cap, tx_vol, ma_window=90)
        assert len(result) == n
        assert result.iloc[:89].isna().all()
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_zero_tx_volume(self) -> None:
        """거래량 0 → NaN."""
        market_cap = pd.Series([1e12] * 90)
        tx_vol = pd.Series([0.0] * 90)
        result = nvt_signal(market_cap, tx_vol, ma_window=90)
        assert pd.isna(result.iloc[-1])


class TestMvrvZscore:
    """mvrv_zscore 테스트."""

    def test_basic(self, sample_index: pd.DatetimeIndex) -> None:
        """z-score 계산 확인."""
        np.random.seed(42)
        mvrv = pd.Series(
            1.5 + np.random.randn(len(sample_index)) * 0.3,
            index=sample_index,
        )
        result = mvrv_zscore(mvrv, window=365)

        valid = result.dropna()
        assert len(valid) > 0
        # z-score 중앙값 합리적 범위
        assert valid.abs().median() < 3.0

    def test_constant_series(self) -> None:
        """상수 MVRV → std=0 → NaN."""
        mvrv = pd.Series([2.0] * 400)
        result = mvrv_zscore(mvrv, window=365)
        valid = result.dropna()
        assert len(valid) == 0


class TestExchangeFlowNetZscore:
    """exchange_flow_net_zscore 테스트."""

    def test_basic(self) -> None:
        """z-score 계산 확인."""
        np.random.seed(42)
        n = 100
        flow_in = pd.Series(np.random.uniform(1e8, 5e8, n))
        flow_out = pd.Series(np.random.uniform(1e8, 5e8, n))

        result = exchange_flow_net_zscore(flow_in, flow_out, window=30)
        valid = result.dropna()
        assert len(valid) > 0
        assert valid.abs().median() < 3.0

    def test_equal_flows(self) -> None:
        """동일 입출금 → net=0 → z-score 계산 가능."""
        flow = pd.Series([1e8] * 50)
        result = exchange_flow_net_zscore(flow, flow, window=30)
        # 상수 net flow → std=0 → NaN
        valid = result.dropna()
        assert len(valid) == 0


class TestStablecoinSupplyRatio:
    """stablecoin_supply_ratio 테스트."""

    def test_basic(self) -> None:
        """SSR 기본 계산."""
        btc_cap = pd.Series([1e12, 1.2e12, 0.8e12])
        stable = pd.Series([1e11, 1.2e11, 1e11])

        result = stablecoin_supply_ratio(btc_cap, stable)
        np.testing.assert_almost_equal(result.iloc[0], 10.0)
        np.testing.assert_almost_equal(result.iloc[1], 10.0)
        np.testing.assert_almost_equal(result.iloc[2], 8.0)

    def test_zero_supply(self) -> None:
        """공급량 0 → NaN."""
        btc_cap = pd.Series([1e12])
        stable = pd.Series([0.0])
        result = stablecoin_supply_ratio(btc_cap, stable)
        assert pd.isna(result.iloc[0])


class TestTvlStablecoinRatio:
    """tvl_stablecoin_ratio 테스트."""

    def test_basic(self) -> None:
        """비율 기본 계산."""
        tvl = pd.Series([5e10, 8e10])
        stable = pd.Series([1e11, 1e11])
        result = tvl_stablecoin_ratio(tvl, stable)
        np.testing.assert_almost_equal(result.iloc[0], 0.5)
        np.testing.assert_almost_equal(result.iloc[1], 0.8)

    def test_zero_supply(self) -> None:
        """공급량 0 → NaN."""
        tvl = pd.Series([5e10])
        stable = pd.Series([0.0])
        result = tvl_stablecoin_ratio(tvl, stable)
        assert pd.isna(result.iloc[0])
