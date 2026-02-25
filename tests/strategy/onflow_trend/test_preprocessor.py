"""Tests for OnFlow Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.onflow_trend.config import OnflowTrendConfig
from src.strategy.onflow_trend.preprocessor import preprocess


@pytest.fixture
def config() -> OnflowTrendConfig:
    return OnflowTrendConfig()


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
def sample_ohlcv_with_onchain(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + On-chain 컬럼 포함 테스트 데이터."""
    df = sample_ohlcv_df.copy()
    np.random.seed(123)
    n = len(df)
    df["oc_flow_in_ex_usd"] = np.random.uniform(1e6, 1e9, n)
    df["oc_flow_out_ex_usd"] = np.random.uniform(1e6, 1e9, n)
    df["oc_mvrv"] = np.random.uniform(0.5, 4.0, n)
    return df


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "netflow_zscore",
            "mvrv_conviction",
            "ema_fast",
            "ema_slow",
            "trend_up",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: OnflowTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_trend_up_binary(
        self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["trend_up"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_with_onchain_columns(
        self, sample_ohlcv_with_onchain: pd.DataFrame, config: OnflowTrendConfig
    ) -> None:
        """On-chain 컬럼 존재 시 실제 feature 계산."""
        result = preprocess(sample_ohlcv_with_onchain, config)
        # netflow_zscore는 0이 아닌 값 포함
        valid = result["netflow_zscore"].dropna()
        assert (valid != 0).any()
        # mvrv_conviction은 0.3, 1.0, 1.2 중 하나
        valid_mvrv = result["mvrv_conviction"].dropna()
        assert set(valid_mvrv.unique()).issubset({0.3, 1.0, 1.2})


class TestGracefulDegradation:
    """On-chain 컬럼 부재 시 Graceful Degradation."""

    def test_without_onchain_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig
    ) -> None:
        """oc_* 컬럼 없이도 에러 없이 실행."""
        result = preprocess(sample_ohlcv_df, config)
        assert "netflow_zscore" in result.columns
        assert "mvrv_conviction" in result.columns

    def test_netflow_zero_without_onchain(
        self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig
    ) -> None:
        """On-chain 부재 시 netflow_zscore = 0."""
        result = preprocess(sample_ohlcv_df, config)
        assert (result["netflow_zscore"] == 0.0).all()

    def test_mvrv_neutral_without_onchain(
        self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig
    ) -> None:
        """On-chain 부재 시 mvrv_conviction = 1.0."""
        result = preprocess(sample_ohlcv_df, config)
        assert (result["mvrv_conviction"] == 1.0).all()

    def test_partial_onchain(
        self, sample_ohlcv_with_onchain: pd.DataFrame, config: OnflowTrendConfig
    ) -> None:
        """일부 oc_* 컬럼만 있어도 정상 동작."""
        df = sample_ohlcv_with_onchain.drop(columns=["oc_mvrv"])
        result = preprocess(df, config)
        assert "mvrv_conviction" in result.columns
        assert (result["mvrv_conviction"] == 1.0).all()
        # flow는 정상 계산
        valid = result["netflow_zscore"].dropna()
        assert (valid != 0).any()


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: OnflowTrendConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
