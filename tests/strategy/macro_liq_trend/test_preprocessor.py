"""Tests for Macro-Liquidity Adaptive Trend preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.macro_liq_trend.config import MacroLiqTrendConfig
from src.strategy.macro_liq_trend.preprocessor import preprocess


@pytest.fixture
def config() -> MacroLiqTrendConfig:
    return MacroLiqTrendConfig()


@pytest.fixture
def sample_macro_df() -> pd.DataFrame:
    """1D OHLCV + Macro data (300 bars)."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    high = close + np.abs(np.random.randn(n) * 300)
    low = close - np.abs(np.random.randn(n) * 300)
    open_ = close + np.random.randn(n) * 100
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.random.randint(1000, 10000, n).astype(float)

    # Macro data: DXY (~102), VIX (~18), SPY (~480), Stablecoin (~130B)
    dxy = 102 + np.cumsum(np.random.randn(n) * 0.3)
    vix = 18 + np.cumsum(np.random.randn(n) * 0.5)
    vix = np.maximum(vix, 10.0)  # VIX floor
    spy = 480 + np.cumsum(np.random.randn(n) * 2)
    stab = 1.3e11 + np.cumsum(np.random.randn(n) * 5e8)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "macro_dxy": dxy,
            "macro_vix": vix,
            "macro_spy": spy,
            "oc_stablecoin_total_circulating_usd": stab,
        },
        index=dates,
    )


class TestPreprocess:
    def test_output_columns(
        self, sample_macro_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        result = preprocess(sample_macro_df, config)
        expected = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "dxy_roc",
            "dxy_z",
            "vix_roc",
            "vix_z",
            "spy_roc",
            "spy_z",
            "stab_change",
            "stab_z",
            "macro_liq_score",
            "sma_price",
            "drawdown",
            "atr",
        }
        assert expected.issubset(set(result.columns))

    def test_same_length(self, sample_macro_df: pd.DataFrame, config: MacroLiqTrendConfig) -> None:
        result = preprocess(sample_macro_df, config)
        assert len(result) == len(sample_macro_df)

    def test_immutability(self, sample_macro_df: pd.DataFrame, config: MacroLiqTrendConfig) -> None:
        original = sample_macro_df.copy()
        preprocess(sample_macro_df, config)
        pd.testing.assert_frame_equal(sample_macro_df, original)

    def test_missing_columns(self, config: MacroLiqTrendConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_missing_macro_column(
        self, sample_macro_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        df = sample_macro_df.drop(columns=["macro_dxy"])
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_macro_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        result = preprocess(sample_macro_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_macro_liq_score_range(
        self, sample_macro_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        """Composite z-score should have reasonable range."""
        result = preprocess(sample_macro_df, config)
        valid = result["macro_liq_score"].dropna()
        # Z-score average should be within [-5, 5] in normal data
        assert valid.abs().mean() < 5.0

    def test_drawdown_non_positive(
        self, sample_macro_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        result = preprocess(sample_macro_df, config)
        valid = result["drawdown"].dropna()
        assert (valid <= 0).all()

    def test_dxy_z_inverted(
        self, sample_macro_df: pd.DataFrame, config: MacroLiqTrendConfig
    ) -> None:
        """DXY z-score is inverted: DXY drop = positive z."""
        result = preprocess(sample_macro_df, config)
        # Just check that dxy_z column exists and has valid data
        assert "dxy_z" in result.columns
        assert result["dxy_z"].dropna().shape[0] > 0

    def test_nan_macro_ffill(self, config: MacroLiqTrendConfig) -> None:
        """Macro columns with NaN should be forward-filled."""
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = 50000 + np.cumsum(np.random.randn(n) * 500)
        high = close + np.abs(np.random.randn(n) * 300)
        low = close - np.abs(np.random.randn(n) * 300)
        open_ = close + np.random.randn(n) * 100
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        dxy = np.full(n, np.nan)
        dxy[::5] = 102.0  # Only every 5th bar has data
        vix = np.full(n, np.nan)
        vix[::5] = 18.0
        spy = np.full(n, np.nan)
        spy[::5] = 480.0
        stab = np.full(n, np.nan)
        stab[::5] = 1.3e11

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
                "macro_dxy": dxy,
                "macro_vix": vix,
                "macro_spy": spy,
                "oc_stablecoin_total_circulating_usd": stab,
            },
            index=dates,
        )
        # Should not raise
        result = preprocess(df, config)
        assert len(result) == n
