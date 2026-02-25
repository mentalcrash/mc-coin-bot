"""Tests for MVRV Cycle Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mvrv_cycle_trend.config import MvrvCycleTrendConfig
from src.strategy.mvrv_cycle_trend.preprocessor import preprocess


@pytest.fixture
def config() -> MvrvCycleTrendConfig:
    return MvrvCycleTrendConfig()


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
def sample_ohlcv_with_onchain(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + On-chain MVRV 포함 테스트 데이터."""
    df = sample_ohlcv_df.copy()
    np.random.seed(42)
    n = len(df)
    df["oc_mvrv"] = np.random.uniform(0.5, 4.0, n)
    return df


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "mom_fast",
            "mom_slow",
            "mom_blend",
            "mvrv_zscore",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: MvrvCycleTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_mom_blend_weighted(
        self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        """mom_blend = weight * mom_fast + (1-weight) * mom_slow."""
        result = preprocess(sample_ohlcv_df, config)
        expected = (
            config.mom_blend_weight * result["mom_fast"]
            + (1.0 - config.mom_blend_weight) * result["mom_slow"]
        )
        pd.testing.assert_series_equal(result["mom_blend"], expected, check_names=False)

    def test_drawdown_nonpositive(
        self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df


class TestGracefulDegradation:
    """On-chain 컬럼 부재 시 Graceful Degradation 검증."""

    def test_without_onchain_mvrv(
        self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        """oc_mvrv 컬럼 없이도 에러 없이 실행."""
        result = preprocess(sample_ohlcv_df, config)
        assert "mvrv_zscore" in result.columns
        # 전부 NaN (graceful degradation)
        assert result["mvrv_zscore"].isna().all()

    def test_with_onchain_mvrv(
        self, sample_ohlcv_with_onchain: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        """oc_mvrv 컬럼 있을 때 mvrv_zscore 계산."""
        result = preprocess(sample_ohlcv_with_onchain, config)
        assert "mvrv_zscore" in result.columns
        # mvrv_zscore_window=365 > 300 rows이므로 대부분 NaN일 수 있음
        # 최소한 에러 없이 실행되면 OK
        assert len(result) == len(sample_ohlcv_with_onchain)

    def test_with_short_mvrv_window(self, sample_ohlcv_with_onchain: pd.DataFrame) -> None:
        """짧은 mvrv_zscore_window로 계산 가능 확인."""
        config = MvrvCycleTrendConfig(mvrv_zscore_window=90)
        result = preprocess(sample_ohlcv_with_onchain, config)
        non_nan = result["mvrv_zscore"].dropna()
        assert len(non_nan) > 0
