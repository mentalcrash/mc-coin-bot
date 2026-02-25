"""Tests for Composite Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.comp_mom.config import CompMomConfig
from src.strategy.comp_mom.preprocessor import preprocess


@pytest.fixture
def config() -> CompMomConfig:
    return CompMomConfig()


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


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "mom_zscore",
            "vol_zscore",
            "gk_zscore",
            "composite_score",
            "drawdown",
            "atr",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: CompMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_composite_score_is_product(
        self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig
    ) -> None:
        """composite_score = mom_zscore * |vol_zscore| * |gk_zscore|."""
        result = preprocess(sample_ohlcv_df, config)
        expected = result["mom_zscore"] * result["vol_zscore"].abs() * result["gk_zscore"].abs()
        valid_mask = result["composite_score"].notna() & expected.notna()
        pd.testing.assert_series_equal(
            result["composite_score"][valid_mask],
            expected[valid_mask],
            check_names=False,
        )

    def test_mom_zscore_has_both_signs(
        self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig
    ) -> None:
        """모멘텀 z-score는 양수/음수 모두 존재해야 함."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["mom_zscore"].dropna()
        assert (valid > 0).any()
        assert (valid < 0).any()

    def test_vol_zscore_mostly_positive_or_zero(
        self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig
    ) -> None:
        """거래량은 항상 양수이므로 z-score 범위가 유한해야 함."""
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_zscore"].dropna()
        assert np.isfinite(valid.values).all()

    def test_drawdown_nonpositive(
        self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_custom_config(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CompMomConfig(mom_period=10, gk_window=10)
        result = preprocess(sample_ohlcv_df, config)
        assert "composite_score" in result.columns


class TestPreprocessorImmutability:
    def test_original_unchanged(self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
