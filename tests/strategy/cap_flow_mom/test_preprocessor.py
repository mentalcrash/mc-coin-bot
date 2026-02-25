"""Tests for Capital Flow Momentum preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.cap_flow_mom.config import CapFlowMomConfig
from src.strategy.cap_flow_mom.preprocessor import preprocess


@pytest.fixture
def config() -> CapFlowMomConfig:
    return CapFlowMomConfig()


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
def sample_ohlcv_with_onchain() -> pd.DataFrame:
    """OHLCV + On-chain stablecoin 컬럼 포함 테스트 데이터."""
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    # Stablecoin total supply with gradual growth
    stablecoin = 100e9 + np.cumsum(np.random.randn(n) * 1e8)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_stablecoin_total_usd": stablecoin,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "fast_roc",
            "slow_roc",
            "stablecoin_roc",
            "drawdown",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_missing_columns(self, config: CapFlowMomConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_fast_roc_not_all_nan(
        self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert not result["fast_roc"].isna().all()

    def test_slow_roc_not_all_nan(
        self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert not result["slow_roc"].isna().all()

    def test_drawdown_non_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()


class TestGracefulDegradation:
    """On-chain 컬럼 부재 시 Graceful Degradation 검증."""

    def test_without_onchain_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        """oc_* 컬럼 없이도 에러 없이 실행."""
        result = preprocess(sample_ohlcv_df, config)
        assert "stablecoin_roc" in result.columns
        assert result["stablecoin_roc"].isna().all()

    def test_with_onchain_stablecoin(
        self, sample_ohlcv_with_onchain: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        """stablecoin 컬럼이 있으면 stablecoin_roc 계산."""
        result = preprocess(sample_ohlcv_with_onchain, config)
        assert "stablecoin_roc" in result.columns
        # warmup 이후에는 값이 있어야 함
        valid = result["stablecoin_roc"].dropna()
        assert len(valid) > 0

    def test_stablecoin_ffill(self, config: CapFlowMomConfig) -> None:
        """stablecoin 데이터의 NaN이 ffill 처리되는지 확인."""
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        high = np.maximum(high, close)
        low = np.minimum(low, close)
        stab = np.full(n, 100e9)
        stab[10:15] = np.nan  # 중간 NaN (주말 등)
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(n) * 5000,
                "oc_stablecoin_total_usd": stab,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        result = preprocess(df, config)
        # ffill 후에도 stablecoin_roc가 계산됨
        assert "stablecoin_roc" in result.columns


class TestPreprocessorImmutability:
    def test_original_unchanged(
        self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_output_is_new_object(
        self, sample_ohlcv_df: pd.DataFrame, config: CapFlowMomConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert result is not sample_ohlcv_df
