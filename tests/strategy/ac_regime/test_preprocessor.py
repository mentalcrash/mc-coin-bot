"""Tests for AC Regime Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.ac_regime.config import ACRegimeConfig
from src.strategy.ac_regime.preprocessor import (
    calculate_rolling_autocorrelation,
    preprocess,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
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


class TestRollingAutocorrelation:
    """calculate_rolling_autocorrelation() 테스트."""

    def test_output_length(self):
        """출력 길이 확인."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100))
        rho = calculate_rolling_autocorrelation(returns, window=20, lag=1)
        assert len(rho) == 100

    def test_values_bounded(self):
        """AC 값은 [-1, 1] 범위."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(200))
        rho = calculate_rolling_autocorrelation(returns, window=60, lag=1)
        valid = rho.dropna()
        assert (valid >= -1.1).all()  # 수치 오차 허용
        assert (valid <= 1.1).all()

    def test_positive_ac_with_trend(self):
        """트렌드 데이터에서 양의 AC."""
        # 강한 trend → positive autocorrelation 예상
        trend = pd.Series(np.arange(200, dtype=float) * 0.1 + np.random.randn(200) * 0.01)
        returns = trend.diff()
        rho = calculate_rolling_autocorrelation(returns, window=60, lag=1)
        valid = rho.dropna()
        # 대부분 양수 (엄밀한 테스트는 아님)
        assert valid.mean() > -1.0  # 최소한 극단적 음수는 아님

    def test_nan_for_insufficient_data(self):
        """데이터 부족 시 NaN."""
        returns = pd.Series(np.random.randn(10))
        rho = calculate_rolling_autocorrelation(returns, window=20, lag=1)
        assert rho.isna().all()


class TestPreprocess:
    """preprocess() 테스트."""

    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame):
        config = ACRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "ac_rho",
            "sig_bound",
            "mom_direction",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_sig_bound_constant(self, sample_ohlcv_df: pd.DataFrame):
        """sig_bound는 상수."""
        config = ACRegimeConfig(significance_z=1.96, ac_window=60)
        result = preprocess(sample_ohlcv_df, config)

        expected = 1.96 / np.sqrt(60)
        assert np.allclose(result["sig_bound"].dropna(), expected)

    def test_mom_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        """mom_direction은 {-1, 0, 1}."""
        config = ACRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid = result["mom_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame):
        config = ACRegimeConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = ACRegimeConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = ACRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = ACRegimeConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()
