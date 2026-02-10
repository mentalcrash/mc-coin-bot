"""Tests for Volume Climax Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.vol_climax.config import VolClimaxConfig
from src.strategy.vol_climax.preprocessor import preprocess


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    # Volume with occasional spikes to simulate climax events
    volume = np.random.randint(1000, 10000, n).astype(float)
    # Insert a few volume spikes
    volume[50] = 50000.0
    volume[100] = 60000.0
    volume[200] = 45000.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestPreprocess:
    def test_output_columns(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig()
        result = preprocess(sample_ohlcv_df, config)

        expected_cols = [
            "returns",
            "volume_zscore",
            "obv",
            "obv_direction",
            "price_direction",
            "divergence",
            "close_position",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_volume_zscore_detects_spikes(self, sample_ohlcv_df: pd.DataFrame):
        """Volume spikes should produce high Z-scores."""
        config = VolClimaxConfig()
        result = preprocess(sample_ohlcv_df, config)

        valid_z = result["volume_zscore"].dropna()
        assert len(valid_z) > 0
        # At least some bars should have high z-score due to injected spikes
        assert valid_z.max() > 2.0

    def test_obv_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["obv_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_price_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["price_direction"].dropna()
        assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})

    def test_divergence_is_bool(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert result["divergence"].dtype == bool

    def test_close_position_range(self, sample_ohlcv_df: pd.DataFrame):
        """Close position should be between 0 and 1."""
        config = VolClimaxConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["close_position"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()

    def test_immutability(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig()
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = VolClimaxConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig()
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_vol_scalar_positive(self, sample_ohlcv_df: pd.DataFrame):
        config = VolClimaxConfig()
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_obv_cumulative(self, sample_ohlcv_df: pd.DataFrame):
        """OBV should be a cumulative sum."""
        config = VolClimaxConfig()
        result = preprocess(sample_ohlcv_df, config)
        # OBV should not be all zeros
        obv = result["obv"].dropna()
        assert len(obv) > 0
        assert not (obv == 0).all()
