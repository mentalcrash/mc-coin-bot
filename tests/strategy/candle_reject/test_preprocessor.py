"""Tests for Candle Reject Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.candle_reject.config import CandleRejectConfig
from src.strategy.candle_reject.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """OHLCV 데이터 생성 (rejection wick이 발생할 수 있는 데이터)."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    # 꼬리가 긴 캔들이 나올 수 있도록 high/low 범위 확장
    open_ = close + np.random.randn(n) * 0.5
    # high/low는 반드시 open, close를 포함해야 함 (valid OHLCV)
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max + np.abs(np.random.randn(n) * 3.0)
    low = bar_min - np.abs(np.random.randn(n) * 3.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "upper_wick",
            "lower_wick",
            "body",
            "range_",
            "bull_reject",
            "bear_reject",
            "body_position",
            "volume_zscore",
            "returns",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_rejection_ratio_range(self, sample_ohlcv_df: pd.DataFrame):
        """bull_reject, bear_reject은 0~1 범위."""
        config = CandleRejectConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_bull = result["bull_reject"].dropna()
        valid_bear = result["bear_reject"].dropna()

        assert (valid_bull >= 0).all()
        assert (valid_bull <= 1).all()
        assert (valid_bear >= 0).all()
        assert (valid_bear <= 1).all()

    def test_body_position_range(self, sample_ohlcv_df: pd.DataFrame):
        """body_position은 0~1 범위."""
        config = CandleRejectConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["body_position"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_wick_non_negative(self, sample_ohlcv_df: pd.DataFrame):
        """upper_wick, lower_wick은 항상 >= 0."""
        config = CandleRejectConfig()
        result = preprocess(sample_ohlcv_df, config)

        assert (result["upper_wick"].dropna() >= 0).all()
        assert (result["lower_wick"].dropna() >= 0).all()

    def test_bull_plus_bear_plus_body_eq_range(self, sample_ohlcv_df: pd.DataFrame):
        """upper_wick + lower_wick + body == range (bar anatomy 일관성)."""
        config = CandleRejectConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result.dropna(subset=["range_"])
        total = valid["upper_wick"] + valid["lower_wick"] + valid["body"]
        pd.testing.assert_series_equal(
            total,
            valid["range_"],
            check_names=False,
            atol=1e-10,
        )

    def test_volume_zscore_exists(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["volume_zscore"].dropna()
        assert len(valid) > 0

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = CandleRejectConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_doji_handling(self):
        """range=0 (doji)인 경우 NaN 처리."""
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 105.0, 103.0],
                "low": [100.0, 95.0, 97.0],
                "close": [100.0, 102.0, 101.0],
                "volume": [1000.0, 2000.0, 1500.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="4h"),
        )
        config = CandleRejectConfig()
        result = preprocess(df, config)

        # 첫 번째 bar: open=high=low=close=100 -> range=0 -> NaN
        assert pd.isna(result["bull_reject"].iloc[0])
        assert pd.isna(result["bear_reject"].iloc[0])
