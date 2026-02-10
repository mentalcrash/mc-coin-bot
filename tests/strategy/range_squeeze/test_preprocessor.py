"""Tests for Range Squeeze Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.range_squeeze.config import RangeSqueezeConfig
from src.strategy.range_squeeze.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (200일)."""
    np.random.seed(42)
    n = 200

    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestPreprocess:
    """preprocess() 테스트."""

    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame):
        """출력 DataFrame에 필요한 컬럼이 존재하는지 확인."""
        config = RangeSqueezeConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "daily_range",
            "avg_range",
            "range_ratio",
            "is_nr",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_daily_range_positive(self, sample_ohlcv_df: pd.DataFrame):
        """daily_range는 항상 >= 0."""
        config = RangeSqueezeConfig()
        result = preprocess(sample_ohlcv_df, config)

        assert (result["daily_range"] >= 0).all()

    def test_range_ratio_reasonable(self, sample_ohlcv_df: pd.DataFrame):
        """range_ratio가 합리적인 범위."""
        config = RangeSqueezeConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["range_ratio"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    def test_is_nr_boolean(self, sample_ohlcv_df: pd.DataFrame):
        """is_nr은 boolean 타입."""
        config = RangeSqueezeConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["is_nr"].dropna()
        assert valid.dtype == bool

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame):
        """vol_scalar는 양수."""
        config = RangeSqueezeConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame):
        """원본 DataFrame은 수정되지 않아야 함."""
        config = RangeSqueezeConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)

        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = RangeSqueezeConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        """출력 길이는 입력과 동일."""
        config = RangeSqueezeConfig()
        result = preprocess(sample_ohlcv_df, config)

        assert len(result) == len(sample_ohlcv_df)

    def test_nr_pattern_detection(self):
        """NR 패턴이 올바르게 감지되는지 확인."""
        n = 50
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)

        # 7번째 bar의 range를 최소로 설정
        high[6] = close[6] + 0.01
        low[6] = close[6] - 0.01

        df = pd.DataFrame(
            {
                "open": close + np.random.randn(n) * 0.5,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = RangeSqueezeConfig(nr_period=7)
        result = preprocess(df, config)

        # NR7 검출 확인 (최소한 하나 이상의 NR 존재)
        valid = result["is_nr"].dropna()
        assert valid.any()
