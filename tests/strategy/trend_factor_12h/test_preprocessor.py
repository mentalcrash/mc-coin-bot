"""Tests for Trend Factor Multi-Horizon preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.trend_factor_12h.config import TrendFactorConfig
from src.strategy.trend_factor_12h.preprocessor import preprocess


@pytest.fixture
def config() -> TrendFactorConfig:
    return TrendFactorConfig()


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
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "tf_h5",
            "tf_h10",
            "tf_h20",
            "tf_h40",
            "tf_h80",
            "trend_factor",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: TrendFactorConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_trend_factor_is_sum_of_components(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        """trend_factor == sum of tf_h{N} components."""
        result = preprocess(sample_ohlcv_df, config)
        horizons = [
            config.horizon_1,
            config.horizon_2,
            config.horizon_3,
            config.horizon_4,
            config.horizon_5,
        ]
        component_df = pd.concat([result[f"tf_h{h}"] for h in horizons], axis=1)
        component_sum = component_df.sum(axis=1)
        pd.testing.assert_series_equal(
            result["trend_factor"],
            component_sum,
            check_names=False,
        )

    def test_tf_components_have_nan_warmup(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        """각 tf_h{N}은 horizon-1 bar 동안 NaN이어야 한다."""
        result = preprocess(sample_ohlcv_df, config)
        # tf_h80 needs at least 80 bars + 1 (returns shift) = 81 NaN
        assert result["tf_h80"].iloc[:80].isna().all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_custom_horizons(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """커스텀 horizon 파라미터로 preprocessor 동작 확인."""
        config = TrendFactorConfig(
            horizon_1=3,
            horizon_2=6,
            horizon_3=12,
            horizon_4=24,
            horizon_5=48,
        )
        result = preprocess(sample_ohlcv_df, config)
        assert "tf_h3" in result.columns
        assert "tf_h6" in result.columns
        assert "tf_h12" in result.columns
        assert "tf_h24" in result.columns
        assert "tf_h48" in result.columns

    def test_returns_are_log_returns(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        expected = np.log(sample_ohlcv_df["close"] / sample_ohlcv_df["close"].shift(1))
        pd.testing.assert_series_equal(
            result["returns"],
            expected,
            check_names=False,
        )


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
