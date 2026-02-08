"""Tests for Stochastic Momentum Hybrid Preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.stoch_mom.config import StochMomConfig
from src.strategy.stoch_mom.preprocessor import preprocess


class TestPreprocess:
    """Preprocessing main function tests."""

    def test_preprocess_adds_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        """All required indicator columns are added after preprocessing."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)

        expected_cols = [
            "pct_k",
            "pct_d",
            "sma",
            "atr",
            "returns",
            "realized_vol",
            "vol_scalar",
            "vol_ratio",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, sample_ohlcv: pd.DataFrame) -> None:
        """Original OHLCV columns are preserved."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)

        for col in ["open", "high", "low", "close"]:
            assert col in result.columns, f"Original column {col} missing"

    def test_original_not_modified(self, sample_ohlcv: pd.DataFrame) -> None:
        """Original DataFrame is not modified."""
        config = StochMomConfig()
        original_cols = list(sample_ohlcv.columns)
        preprocess(sample_ohlcv, config)
        assert list(sample_ohlcv.columns) == original_cols

    def test_output_length(self, sample_ohlcv: pd.DataFrame) -> None:
        """Output length matches input."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)
        assert len(result) == len(sample_ohlcv)

    def test_pct_k_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """Stochastic %K values are in 0-100 range (after warmup)."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)

        k_valid = result["pct_k"].dropna()
        assert len(k_valid) > 0, "No valid %K values"
        assert (k_valid >= 0).all(), f"%K below 0: {k_valid.min()}"
        assert (k_valid <= 100).all(), f"%K above 100: {k_valid.max()}"

    def test_pct_d_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """Stochastic %D values are in 0-100 range (after warmup)."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)

        d_valid = result["pct_d"].dropna()
        assert len(d_valid) > 0, "No valid %D values"
        assert (d_valid >= 0).all(), f"%D below 0: {d_valid.min()}"
        assert (d_valid <= 100).all(), f"%D above 100: {d_valid.max()}"

    def test_vol_scalar_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_scalar is always positive."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR is always positive."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["atr"].dropna()
        assert (valid > 0).all()

    def test_vol_ratio_clipped(self, sample_ohlcv: pd.DataFrame) -> None:
        """vol_ratio is clipped to [min_vol_ratio, max_vol_ratio]."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["vol_ratio"].dropna()
        assert (valid >= config.min_vol_ratio).all(), f"vol_ratio below min: {valid.min()}"
        assert (valid <= config.max_vol_ratio).all(), f"vol_ratio above max: {valid.max()}"

    def test_sma_present(self, sample_ohlcv: pd.DataFrame) -> None:
        """SMA column is present and has valid values."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)
        valid = result["sma"].dropna()
        assert len(valid) > 0

    def test_no_nan_after_warmup(self, sample_ohlcv: pd.DataFrame) -> None:
        """No NaN in key columns after warmup period."""
        config = StochMomConfig()
        result = preprocess(sample_ohlcv, config)

        warmup = config.warmup_periods()
        after_warmup = result.iloc[warmup:]
        check_cols = ["pct_k", "pct_d", "sma", "atr", "vol_scalar", "vol_ratio"]
        for col in check_cols:
            nan_count = after_warmup[col].isna().sum()
            assert nan_count == 0, f"Column {col} has {nan_count} NaNs after warmup"

    def test_missing_columns_raises(self) -> None:
        """Missing required columns raise ValueError."""
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        config = StochMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_single_column_raises(self) -> None:
        """Missing a single required column raises ValueError."""
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame(
            {
                "open": np.random.randn(n),
                "high": np.random.randn(n),
                # low missing
                "close": np.random.randn(n),
            },
            index=dates,
        )
        config = StochMomConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)


class TestPreprocessWithDifferentConfigs:
    """Preprocessing with different configurations."""

    def test_different_k_period(self, sample_ohlcv: pd.DataFrame) -> None:
        """Different k_period works correctly."""
        config = StochMomConfig(k_period=20)
        result = preprocess(sample_ohlcv, config)
        assert "pct_k" in result.columns

    def test_different_sma_period(self, sample_ohlcv: pd.DataFrame) -> None:
        """Different sma_period works correctly."""
        config = StochMomConfig(sma_period=50)
        result = preprocess(sample_ohlcv, config)
        assert "sma" in result.columns

    def test_higher_vol_target_higher_scalar(self, sample_ohlcv: pd.DataFrame) -> None:
        """Higher vol_target produces higher vol_scalar."""
        config_low = StochMomConfig(vol_target=0.10)
        config_high = StochMomConfig(vol_target=0.50)

        result_low = preprocess(sample_ohlcv, config_low)
        result_high = preprocess(sample_ohlcv, config_high)

        avg_low = result_low["vol_scalar"].dropna().mean()
        avg_high = result_high["vol_scalar"].dropna().mean()
        assert avg_high > avg_low
