"""Tests for VRP-Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vrp_trend.config import VrpTrendConfig
from src.strategy.vrp_trend.preprocessor import preprocess


@pytest.fixture
def config() -> VrpTrendConfig:
    return VrpTrendConfig()


@pytest.fixture
def sample_ohlcv_dvol_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    # DVOL: 30~90% range (annualized implied vol)
    dvol = 50.0 + np.cumsum(np.random.randn(n) * 2)
    dvol = np.clip(dvol, 20, 120)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "dvol": dvol,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_dvol_df: pd.DataFrame, config: VrpTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_dvol_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "dvol_clean",
            "rv_annualized_pct",
            "vrp",
            "vrp_ma",
            "vrp_zscore",
            "trend_sma",
            "above_trend",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_dvol_df: pd.DataFrame, config: VrpTrendConfig) -> None:
        result = preprocess(sample_ohlcv_dvol_df, config)
        assert len(result) == len(sample_ohlcv_dvol_df)

    def test_immutability(self, sample_ohlcv_dvol_df: pd.DataFrame, config: VrpTrendConfig) -> None:
        original = sample_ohlcv_dvol_df.copy()
        preprocess(sample_ohlcv_dvol_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_dvol_df, original)

    def test_missing_columns(self, config: VrpTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_dvol_column(self, config: VrpTrendConfig) -> None:
        """OHLCV만 있고 dvol이 없으면 에러."""
        np.random.seed(42)
        n = 50
        df = pd.DataFrame(
            {
                "open": np.random.randn(n) + 100,
                "high": np.random.randn(n) + 101,
                "low": np.random.randn(n) + 99,
                "close": np.random.randn(n) + 100,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_dvol_df: pd.DataFrame, config: VrpTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_dvol_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_vrp_is_iv_minus_rv(
        self, sample_ohlcv_dvol_df: pd.DataFrame, config: VrpTrendConfig
    ) -> None:
        """VRP = DVOL - RV (annualized pct)."""
        result = preprocess(sample_ohlcv_dvol_df, config)
        valid_idx = result["vrp"].dropna().index
        expected = result.loc[valid_idx, "dvol_clean"] - result.loc[valid_idx, "rv_annualized_pct"]
        pd.testing.assert_series_equal(result.loc[valid_idx, "vrp"], expected, check_names=False)

    def test_above_trend_binary(
        self, sample_ohlcv_dvol_df: pd.DataFrame, config: VrpTrendConfig
    ) -> None:
        """above_trend is 0 or 1."""
        result = preprocess(sample_ohlcv_dvol_df, config)
        valid = result["above_trend"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_dvol_ffill(self, config: VrpTrendConfig) -> None:
        """DVOL에 NaN이 있어도 ffill로 처리."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)
        open_ = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 10000, n).astype(float)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        dvol = np.full(n, 55.0)
        dvol[::3] = np.nan  # 매 3번째 NaN
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "dvol": dvol,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        result = preprocess(df, config)
        # dvol_clean에는 첫 행(NaN) 제외하고 NaN이 없어야 함
        assert result["dvol_clean"].iloc[1:].notna().all()

    def test_drawdown_range(
        self, sample_ohlcv_dvol_df: pd.DataFrame, config: VrpTrendConfig
    ) -> None:
        """drawdown은 0 이하."""
        result = preprocess(sample_ohlcv_dvol_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()
