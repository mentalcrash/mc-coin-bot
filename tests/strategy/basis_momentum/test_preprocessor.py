"""Tests for Basis-Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.basis_momentum.config import BasisMomentumConfig
from src.strategy.basis_momentum.preprocessor import preprocess


@pytest.fixture
def config() -> BasisMomentumConfig:
    return BasisMomentumConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


@pytest.fixture
def sample_ohlcv_with_fr(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_ohlcv_df.copy()
    np.random.seed(123)
    # Synthetic funding rate: positive mean (bullish bias) with noise
    df["funding_rate"] = 0.001 + np.random.randn(len(df)) * 0.0005
    return df


class TestPreprocessColumns:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        """전처리 후 FR + vol 컬럼이 추가되는지 확인."""
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "fr_change",
            "fr_change_std",
            "basis_mom",
            "returns",
            "realized_vol",
            "vol_scalar",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_df: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: BasisMomentumConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)


class TestGracefulDegradation:
    def test_no_funding_rate_basis_mom_all_zero(
        self, sample_ohlcv_df: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        """funding_rate 컬럼 부재 시 basis_mom = 0 (crash 없음)."""
        result = preprocess(sample_ohlcv_df, config)
        assert (result["basis_mom"] == 0.0).all()

    def test_with_funding_rate_basis_mom_computed(
        self, sample_ohlcv_with_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        """funding_rate 존재 시 basis_mom이 계산됨."""
        result = preprocess(sample_ohlcv_with_fr, config)
        valid = result["basis_mom"].dropna()
        # FR이 있으므로 0이 아닌 값이 존재해야 함
        assert (valid != 0.0).any(), "basis_mom should have non-zero values with FR"


class TestVolFeatures:
    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_drawdown_lte_zero(
        self, sample_ohlcv_df: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_preserves_original_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns
