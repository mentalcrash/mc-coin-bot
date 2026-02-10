"""Tests for Permutation Entropy Momentum Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.perm_entropy_mom.config import PermEntropyMomConfig
from src.strategy.perm_entropy_mom.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
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
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "pe_short",
            "pe_long",
            "mom_direction",
            "realized_vol",
            "vol_scalar",
            "conviction",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = PermEntropyMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_pe_short_range_0_to_1(self, sample_ohlcv_df: pd.DataFrame):
        """PE values should be in [0, 1] range."""
        config = PermEntropyMomConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_pe = result["pe_short"].dropna()
        assert len(valid_pe) > 0
        assert (valid_pe >= 0).all()
        assert (valid_pe <= 1.0 + 1e-10).all()

    def test_pe_long_range_0_to_1(self, sample_ohlcv_df: pd.DataFrame):
        """PE values should be in [0, 1] range."""
        config = PermEntropyMomConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_pe = result["pe_long"].dropna()
        assert len(valid_pe) > 0
        assert (valid_pe >= 0).all()
        assert (valid_pe <= 1.0 + 1e-10).all()

    def test_conviction_range(self, sample_ohlcv_df: pd.DataFrame):
        """Conviction = 1 - PE, should be in [0, 1]."""
        config = PermEntropyMomConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_conv = result["conviction"].dropna()
        assert len(valid_conv) > 0
        assert (valid_conv >= -1e-10).all()
        assert (valid_conv <= 1.0 + 1e-10).all()

    def test_conviction_inverse_of_pe(self, sample_ohlcv_df: pd.DataFrame):
        """Conviction should equal 1 - pe_short."""
        config = PermEntropyMomConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_mask = result["pe_short"].notna()
        expected = 1.0 - result["pe_short"][valid_mask]
        actual = result["conviction"][valid_mask]
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_mom_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        """Momentum direction should be {-1, 0, 1}."""
        config = PermEntropyMomConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["mom_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = PermEntropyMomConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_pe_higher_order(self, sample_ohlcv_df: pd.DataFrame):
        """Higher PE order should still produce valid [0, 1] values."""
        config = PermEntropyMomConfig(pe_order=5, pe_short_window=30, pe_long_window=60)
        result = preprocess(sample_ohlcv_df, config)

        valid_pe = result["pe_short"].dropna()
        assert len(valid_pe) > 0
        assert (valid_pe >= 0).all()
        assert (valid_pe <= 1.0 + 1e-10).all()

    def test_random_data_high_pe(self):
        """Pure random data should have high PE (close to 1)."""
        np.random.seed(123)
        n = 200
        # Pure random walk -> high entropy
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + 0.5
        low = close - 0.5
        open_ = close + np.random.randn(n) * 0.1

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        config = PermEntropyMomConfig(pe_short_window=30, pe_long_window=60)
        result = preprocess(df, config)

        valid_pe = result["pe_short"].dropna()
        # Random data should have PE > 0.7 on average
        assert valid_pe.mean() > 0.7

    def test_trending_data_lower_pe(self):
        """Strong monotone trend should have lower PE than random."""
        n = 200
        # Strong uptrend with very small noise
        close = np.linspace(100, 200, n)
        high = close + 0.5
        low = close - 0.5
        open_ = close - 0.2

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        config = PermEntropyMomConfig(pe_short_window=30, pe_long_window=60)
        result = preprocess(df, config)

        valid_pe = result["pe_short"].dropna()
        # Monotone trend has very ordered patterns -> low PE
        assert valid_pe.mean() < 0.5
