"""Tests for OU Mean Reversion Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.ou_meanrev.config import OUMeanRevConfig
from src.strategy.ou_meanrev.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Mean-reverting price series for OU testing."""
    np.random.seed(42)
    n = 300
    # Generate a mean-reverting (OU-like) process
    prices = np.zeros(n)
    prices[0] = 100.0
    mu = 100.0
    theta = 0.05  # mean reversion speed
    sigma = 2.0
    for i in range(1, n):
        prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + sigma * np.random.randn()

    high = prices + np.abs(np.random.randn(n) * 1.5)
    low = prices - np.abs(np.random.randn(n) * 1.5)
    open_ = prices + np.random.randn(n) * 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": prices,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "theta",
            "half_life",
            "ou_mu",
            "ou_zscore",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_theta_positive_for_mr_data(self, sample_ohlcv_df: pd.DataFrame):
        """Mean-reverting 데이터에서 theta는 대부분 양수."""
        config = OUMeanRevConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_theta = result["theta"].dropna()
        if len(valid_theta) > 0:
            # OU process data should yield mostly positive theta
            assert (valid_theta > 0).mean() > 0.5

    def test_half_life_positive(self, sample_ohlcv_df: pd.DataFrame):
        """Half-life는 항상 양수."""
        config = OUMeanRevConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_hl = result["half_life"].dropna()
        if len(valid_hl) > 0:
            assert (valid_hl > 0).all()

    def test_ou_mu_reasonable(self, sample_ohlcv_df: pd.DataFrame):
        """OU mu는 가격 범위 내에 있어야 함."""
        config = OUMeanRevConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_mu = result["ou_mu"].dropna()
        if len(valid_mu) > 0:
            close_min = sample_ohlcv_df["close"].min()
            close_max = sample_ohlcv_df["close"].max()
            # mu should be within a reasonable range of price
            # (allow wide margin for estimation noise)
            mu_median = valid_mu.median()
            assert close_min * 0.5 < mu_median < close_max * 2.0

    def test_ou_zscore_centered(self, sample_ohlcv_df: pd.DataFrame):
        """Z-score는 대략 0 중심이어야 함."""
        config = OUMeanRevConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_z = result["ou_zscore"].dropna()
        if len(valid_z) > 0:
            mean_z = valid_z.mean()
            assert -3.0 < mean_z < 3.0

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = OUMeanRevConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_custom_ou_window(self, sample_ohlcv_df: pd.DataFrame):
        """다른 ou_window에서도 정상 동작."""
        config = OUMeanRevConfig(ou_window=60)
        result = preprocess(sample_ohlcv_df, config)

        # More valid data with smaller window
        valid_hl = result["half_life"].dropna()
        assert len(valid_hl) > 0
