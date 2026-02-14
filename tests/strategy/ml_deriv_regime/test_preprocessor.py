"""Tests for ML Derivatives Regime preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.ml_deriv_regime.config import MlDerivRegimeConfig
from src.strategy.ml_deriv_regime.preprocessor import preprocess


@pytest.fixture
def config() -> MlDerivRegimeConfig:
    return MlDerivRegimeConfig()


@pytest.fixture
def sample_ohlcv_with_funding_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 400
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
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
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        required = {"returns", "realized_vol", "vol_scalar", "forward_return", "drawdown"}
        assert required.issubset(set(result.columns))
        # Check derivatives features
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) >= 10

    def test_same_length(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        assert len(result) == len(sample_ohlcv_with_funding_df)

    def test_immutability(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        original = sample_ohlcv_with_funding_df.copy()
        preprocess(sample_ohlcv_with_funding_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_with_funding_df, original)

    def test_missing_columns(self, config: MlDerivRegimeConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_funding_rate(self, config: MlDerivRegimeConfig) -> None:
        """funding_rate 없으면 에러."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "open": np.random.randn(n),
                "high": np.random.randn(n),
                "low": np.random.randn(n),
                "close": np.random.randn(n),
                "volume": np.random.randn(n),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        result = preprocess(sample_ohlcv_with_funding_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_regime_features_added_when_present(
        self, sample_ohlcv_with_funding_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        """regime 컬럼이 있으면 feat_ 접두사로 추가."""
        df = sample_ohlcv_with_funding_df.copy()
        df["p_trending"] = 0.5
        df["p_ranging"] = 0.3
        df["p_volatile"] = 0.2
        result = preprocess(df, config)
        assert "feat_p_trending" in result.columns
        assert "feat_p_ranging" in result.columns
        assert "feat_p_volatile" in result.columns
