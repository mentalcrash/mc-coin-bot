"""Tests for Overnight Seasonality preprocessor."""

import pandas as pd
import pytest

from src.strategy.overnight.config import OvernightConfig
from src.strategy.overnight.preprocessor import preprocess


class TestPreprocessColumns:
    """Verify preprocessor adds expected columns."""

    def test_preprocess_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """preprocess adds hour, returns, realized_vol, vol_scalar, atr."""
        config = OvernightConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = {"hour", "returns", "realized_vol", "vol_scalar", "atr"}
        assert expected_cols.issubset(set(result.columns))

    def test_hour_column_values(self, sample_ohlcv: pd.DataFrame) -> None:
        """hour column contains values 0-23."""
        config = OvernightConfig()
        result = preprocess(sample_ohlcv, config)

        hour_values = result["hour"].unique()
        assert all(0 <= h <= 23 for h in hour_values)
        # 1H data over ~20 days should have all 24 hours
        assert len(hour_values) == 24

    def test_original_dataframe_not_modified(self, sample_ohlcv: pd.DataFrame) -> None:
        """preprocess returns a copy, not modifying the original."""
        config = OvernightConfig()
        original_cols = set(sample_ohlcv.columns)
        _ = preprocess(sample_ohlcv, config)

        assert set(sample_ohlcv.columns) == original_cols

    def test_returns_are_log_returns(self, sample_ohlcv: pd.DataFrame) -> None:
        """returns column is log returns."""
        config = OvernightConfig()
        result = preprocess(sample_ohlcv, config)

        # Log returns should be small values centered around 0
        valid_returns = result["returns"].dropna()
        assert abs(valid_returns.mean()) < 0.1
        assert valid_returns.std() < 1.0


class TestPreprocessVolFilter:
    """Volatility filter column tests."""

    def test_vol_filter_column(self, sample_ohlcv: pd.DataFrame) -> None:
        """When use_vol_filter=True, rolling_vol_ratio column exists."""
        config = OvernightConfig(use_vol_filter=True)
        result = preprocess(sample_ohlcv, config)

        assert "rolling_vol_ratio" in result.columns
        # rolling_vol_ratio should have valid (non-NaN) values after warmup
        valid_ratio = result["rolling_vol_ratio"].dropna()
        assert len(valid_ratio) > 0
        assert (valid_ratio > 0).all()

    def test_no_vol_filter_column_by_default(self, sample_ohlcv: pd.DataFrame) -> None:
        """When use_vol_filter=False (default), no rolling_vol_ratio column."""
        config = OvernightConfig()
        result = preprocess(sample_ohlcv, config)

        assert "rolling_vol_ratio" not in result.columns


class TestPreprocessValidation:
    """Input validation tests."""

    def test_missing_columns_raises(self) -> None:
        """Missing required OHLCV columns raises ValueError."""
        df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1h"),
        )
        config = OvernightConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_volume_raises(self) -> None:
        """Missing volume column raises ValueError."""
        df = pd.DataFrame(
            {"open": [1], "high": [2], "low": [0.5], "close": [1.5]},
            index=pd.date_range("2024-01-01", periods=1, freq="1h"),
        )
        config = OvernightConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)
