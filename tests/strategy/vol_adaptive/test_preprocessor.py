"""Tests for Vol-Adaptive Trend preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_adaptive.config import VolAdaptiveConfig
from src.strategy.vol_adaptive.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (200일)."""
    np.random.seed(42)
    n = 200

    # 상승 추세 + 노이즈
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )

    return df


@pytest.fixture
def sample_config() -> VolAdaptiveConfig:
    """기본 VolAdaptiveConfig."""
    return VolAdaptiveConfig()


class TestPreprocess:
    """preprocess 함수 테스트."""

    def test_preprocess_returns_all_columns(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolAdaptiveConfig,
    ) -> None:
        """preprocess 출력에 필수 컬럼 존재."""
        result = preprocess(sample_ohlcv_df, sample_config)

        expected_cols = [
            "ema_fast",
            "ema_slow",
            "rsi",
            "adx",
            "vol_scalar",
            "atr",
            "drawdown",
            "returns",
            "realized_vol",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_range(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolAdaptiveConfig,
    ) -> None:
        """RSI는 0~100 범위 내."""
        result = preprocess(sample_ohlcv_df, sample_config)

        rsi = result["rsi"].dropna()
        assert len(rsi) > 0
        assert rsi.min() >= 0.0
        assert rsi.max() <= 100.0

    def test_ema_fast_slower_than_slow(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolAdaptiveConfig,
    ) -> None:
        """ema_fast가 ema_slow보다 가격 변화에 더 빠르게 반응."""
        result = preprocess(sample_ohlcv_df, sample_config)

        ema_fast = result["ema_fast"].dropna()
        ema_slow = result["ema_slow"].dropna()

        # EMA fast가 close에 더 가까움 (잔차의 절대값 합이 작음)
        close = result["close"].dropna()
        common_idx = ema_fast.index.intersection(ema_slow.index).intersection(close.index)

        fast_residual = (ema_fast.loc[common_idx] - close.loc[common_idx]).abs().mean()
        slow_residual = (ema_slow.loc[common_idx] - close.loc[common_idx]).abs().mean()

        assert fast_residual < slow_residual

    def test_adx_nonnegative(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolAdaptiveConfig,
    ) -> None:
        """ADX는 0 이상."""
        result = preprocess(sample_ohlcv_df, sample_config)

        adx = result["adx"].dropna()
        assert len(adx) > 0
        assert (adx >= 0).all()

    def test_missing_columns_raises(
        self,
        sample_config: VolAdaptiveConfig,
    ) -> None:
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, sample_config)

    def test_empty_df_raises(
        self,
        sample_config: VolAdaptiveConfig,
    ) -> None:
        """빈 DataFrame 시 에러."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
        )

        with pytest.raises(ValueError, match="empty"):
            preprocess(df, sample_config)

    def test_original_not_modified(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolAdaptiveConfig,
    ) -> None:
        """원본 DataFrame이 수정되지 않음."""
        original_cols = list(sample_ohlcv_df.columns)
        original_values = sample_ohlcv_df["close"].values.copy()

        _ = preprocess(sample_ohlcv_df, sample_config)

        assert list(sample_ohlcv_df.columns) == original_cols
        np.testing.assert_array_equal(sample_ohlcv_df["close"].values, original_values)

    def test_drawdown_is_non_positive(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: VolAdaptiveConfig,
    ) -> None:
        """drawdown은 항상 0 이하."""
        result = preprocess(sample_ohlcv_df, sample_config)

        drawdown = result["drawdown"].dropna()
        assert (drawdown <= 0).all()
