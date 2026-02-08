"""Tests for Larry Williams Volatility Breakout preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.larry_vb.config import LarryVBConfig
from src.strategy.larry_vb.preprocessor import (
    calculate_breakout_levels,
    calculate_prev_range,
    preprocess,
)


class TestPreprocessColumns:
    """Verify preprocessor adds expected columns."""

    def test_preprocess_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """preprocess adds prev_range, breakout_upper/lower, realized_vol, vol_scalar."""
        config = LarryVBConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = {
            "prev_range",
            "breakout_upper",
            "breakout_lower",
            "realized_vol",
            "vol_scalar",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_original_dataframe_not_modified(self, sample_ohlcv: pd.DataFrame) -> None:
        """preprocess returns a copy, not modifying the original."""
        config = LarryVBConfig()
        original_cols = set(sample_ohlcv.columns)
        _ = preprocess(sample_ohlcv, config)

        assert set(sample_ohlcv.columns) == original_cols

    def test_prev_range_is_shifted(self, sample_ohlcv: pd.DataFrame) -> None:
        """prev_range is (high - low) shifted by 1."""
        config = LarryVBConfig()
        result = preprocess(sample_ohlcv, config)

        # First value should be NaN (shift)
        assert pd.isna(result["prev_range"].iloc[0])

        # Second value should equal first bar's range
        expected = sample_ohlcv["high"].iloc[0] - sample_ohlcv["low"].iloc[0]
        np.testing.assert_almost_equal(result["prev_range"].iloc[1], expected)

    def test_vol_scalar_is_shifted(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_scalar has shift(1) applied in preprocessor."""
        config = LarryVBConfig()
        result = preprocess(sample_ohlcv, config)

        # vol_scalar should have more NaN at the start than realized_vol
        # because of the additional shift(1)
        vol_scalar_first_valid = result["vol_scalar"].first_valid_index()
        realized_vol_first_valid = result["realized_vol"].first_valid_index()
        assert vol_scalar_first_valid > realized_vol_first_valid  # type: ignore[operator]


class TestBreakoutLevels:
    """Breakout level calculation tests."""

    def test_breakout_levels_formula(self) -> None:
        """upper = open + k * prev_range, lower = open - k * prev_range."""
        open_price = pd.Series([100.0, 200.0, 300.0])
        prev_range = pd.Series([10.0, 20.0, 30.0])
        k = 0.5

        upper, lower = calculate_breakout_levels(open_price, prev_range, k)

        np.testing.assert_array_almost_equal(upper.values, [105.0, 210.0, 315.0])
        np.testing.assert_array_almost_equal(lower.values, [95.0, 190.0, 285.0])

    def test_breakout_levels_with_nan(self) -> None:
        """NaN in prev_range propagates to breakout levels."""
        open_price = pd.Series([100.0, 200.0])
        prev_range = pd.Series([np.nan, 20.0])
        k = 0.5

        upper, lower = calculate_breakout_levels(open_price, prev_range, k)

        assert pd.isna(upper.iloc[0])
        assert pd.isna(lower.iloc[0])
        np.testing.assert_almost_equal(upper.iloc[1], 210.0)
        np.testing.assert_almost_equal(lower.iloc[1], 190.0)

    def test_prev_range_always_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """prev_range is always >= 0 (high >= low)."""
        high: pd.Series = sample_ohlcv["high"]  # type: ignore[assignment]
        low: pd.Series = sample_ohlcv["low"]  # type: ignore[assignment]
        prev_range = calculate_prev_range(high, low)
        valid = prev_range.dropna()
        assert (valid >= 0).all()

    def test_breakout_upper_above_lower(self, sample_ohlcv: pd.DataFrame) -> None:
        """breakout_upper is always >= breakout_lower."""
        config = LarryVBConfig()
        result = preprocess(sample_ohlcv, config)

        valid = result.dropna(subset=["breakout_upper", "breakout_lower"])
        assert (valid["breakout_upper"] >= valid["breakout_lower"]).all()


class TestPreprocessValidation:
    """Input validation tests."""

    def test_missing_columns_raises(self) -> None:
        """Missing required OHLC columns raises ValueError."""
        df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        config = LarryVBConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_open_raises(self) -> None:
        """Missing open column raises ValueError."""
        df = pd.DataFrame(
            {"high": [2], "low": [0.5], "close": [1.5]},
            index=pd.date_range("2024-01-01", periods=1, freq="1D"),
        )
        config = LarryVBConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_volume_not_required(self) -> None:
        """Volume is not required for Larry VB."""
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame(
            {
                "open": np.random.randn(n) + 100,
                "high": np.random.randn(n) + 102,
                "low": np.random.randn(n) + 98,
                "close": np.random.randn(n) + 100,
            },
            index=dates,
        )
        config = LarryVBConfig()
        result = preprocess(df, config)

        assert "prev_range" in result.columns
