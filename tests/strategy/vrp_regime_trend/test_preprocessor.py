"""Tests for VRP-Regime Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vrp_regime_trend.config import VrpRegimeTrendConfig
from src.strategy.vrp_regime_trend.preprocessor import preprocess


@pytest.fixture
def config() -> VrpRegimeTrendConfig:
    return VrpRegimeTrendConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="8h"),
    )


@pytest.fixture
def sample_ohlcv_with_dvol(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + DVOL 포함 테스트 데이터."""
    df = sample_ohlcv_df.copy()
    np.random.seed(123)
    df["opt_dvol"] = np.random.uniform(40, 80, len(df))
    return df


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "gk_rv",
            "gk_rv_ann_pct",
            "dvol_clean",
            "vrp",
            "vrp_ma",
            "vrp_zscore",
            "ema_fast",
            "ema_slow",
            "trend_up",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: VrpRegimeTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_trend_up_binary(
        self, sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["trend_up"].dropna()
        assert set(valid.unique()).issubset({0, 1})


class TestGracefulDegradation:
    """DVOL 컬럼 부재 시 Graceful Degradation."""

    def test_without_dvol(
        self, sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        """opt_dvol 없어도 에러 없이 실행 (VRP ≈ 0)."""
        result = preprocess(sample_ohlcv_df, config)
        assert "dvol_clean" in result.columns
        assert "vrp" in result.columns

    def test_with_dvol(
        self, sample_ohlcv_with_dvol: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        """opt_dvol 있으면 실제 VRP 계산."""
        result = preprocess(sample_ohlcv_with_dvol, config)
        assert "dvol_clean" in result.columns
        # DVOL이 있으므로 VRP는 0이 아닌 값을 가져야 함
        valid_vrp = result["vrp"].dropna()
        assert not (valid_vrp == 0).all()

    def test_vrp_zero_without_dvol(
        self, sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        """DVOL 부재 → dvol_clean = gk_rv_ann_pct → VRP ≈ 0."""
        result = preprocess(sample_ohlcv_df, config)
        # dvol_clean == gk_rv_ann_pct → vrp = 0 (완벽하지는 않지만 거의 0)
        valid_vrp = result["vrp"].dropna()
        assert valid_vrp.abs().max() < 1e-10


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
