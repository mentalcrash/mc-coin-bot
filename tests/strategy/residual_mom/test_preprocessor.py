"""Tests for Residual Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.residual_mom.config import ResidualMomConfig
from src.strategy.residual_mom.preprocessor import preprocess


@pytest.fixture
def config() -> ResidualMomConfig:
    return ResidualMomConfig()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: ResidualMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "market_returns",
            "residual",
            "residual_mom",
            "residual_mom_zscore",
            "residual_vol",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: ResidualMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: ResidualMomConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: ResidualMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: ResidualMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_residual_has_values(
        self, sample_ohlcv_df: pd.DataFrame, config: ResidualMomConfig
    ) -> None:
        """잔차가 warmup 이후 유효값을 가져야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["residual"].dropna()
        assert len(valid) > 0

    def test_residual_mom_zscore_range(
        self, sample_ohlcv_df: pd.DataFrame, config: ResidualMomConfig
    ) -> None:
        """z-score는 극단적으로 크지 않아야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["residual_mom_zscore"].dropna()
        assert len(valid) > 0
        # z-score는 일반적으로 [-5, 5] 범위
        assert (valid.abs() < 10).all()

    def test_drawdown_nonpositive(
        self, sample_ohlcv_df: pd.DataFrame, config: ResidualMomConfig
    ) -> None:
        """drawdown은 항상 <= 0이어야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_residual_vol_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: ResidualMomConfig
    ) -> None:
        """잔차 변동성은 항상 양수여야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["residual_vol"].dropna()
        assert (valid >= 0).all()
