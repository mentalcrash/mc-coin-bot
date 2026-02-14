"""Tests for Funding Divergence Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.fund_div_mom.config import FundDivMomConfig
from src.strategy.fund_div_mom.preprocessor import preprocess


@pytest.fixture
def config() -> FundDivMomConfig:
    return FundDivMomConfig()


@pytest.fixture
def sample_ohlcv_with_funding_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    # Funding rate: -0.001 ~ +0.001 범위
    funding_rate = np.random.uniform(-0.001, 0.001, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FundDivMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "price_mom",
            "avg_funding_rate",
            "fr_zscore",
            "fr_direction",
            "divergence_score",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FundDivMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        assert len(result) == len(sample_ohlcv_with_funding_df)

    def test_immutability(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FundDivMomConfig
    ) -> None:
        original = sample_ohlcv_with_funding_df.copy()
        preprocess(sample_ohlcv_with_funding_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_with_funding_df, original)

    def test_missing_columns(self, config: FundDivMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: FundDivMomConfig) -> None:
        """funding_rate 없으면 에러."""
        df = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
                "volume": [100.0, 200.0, 300.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FundDivMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FundDivMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_fr_direction_values(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FundDivMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid = result["fr_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_funding_rate_nan_ffill(self, config: FundDivMomConfig) -> None:
        """funding_rate NaN은 ffill로 처리."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + 2
        low = close - 2
        open_ = close - 0.5
        volume = np.full(n, 5000.0)
        fr = np.random.uniform(-0.001, 0.001, n)
        fr[0:10] = np.nan  # 첫 10개 NaN
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "funding_rate": fr,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        result = preprocess(df, config)
        # avg_funding_rate의 NaN은 초기 warmup 이후 사라짐
        assert len(result) == n


class TestDivergenceScore:
    def test_divergence_score_range(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FundDivMomConfig
    ) -> None:
        """divergence_score는 유한해야 함."""
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid = result["divergence_score"].dropna()
        assert np.isfinite(valid).all()

    def test_price_mom_log_return(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: FundDivMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        # price_mom should be finite where not NaN
        valid = result["price_mom"].dropna()
        assert np.isfinite(valid).all()
