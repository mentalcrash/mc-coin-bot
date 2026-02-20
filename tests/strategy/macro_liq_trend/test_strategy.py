"""Tests for Macro-Liquidity Adaptive Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.macro_liq_trend.config import MacroLiqTrendConfig
from src.strategy.macro_liq_trend.strategy import MacroLiqTrendStrategy


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

    dxy = 102 + np.cumsum(np.random.randn(n) * 0.3)
    vix = 18 + np.cumsum(np.random.randn(n) * 0.5)
    vix = np.maximum(vix, 10.0)
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


class TestRegistry:
    def test_registered(self) -> None:
        assert "macro-liq-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("macro-liq-trend")
        assert cls is MacroLiqTrendStrategy


class TestMacroLiqTrendStrategy:
    def test_name(self) -> None:
        strategy = MacroLiqTrendStrategy()
        assert strategy.name == "macro-liq-trend"

    def test_required_columns(self) -> None:
        strategy = MacroLiqTrendStrategy()
        cols = strategy.required_columns
        assert "close" in cols
        assert "volume" in cols
        assert "macro_dxy" in cols
        assert "macro_vix" in cols
        assert "macro_spy" in cols
        assert "oc_stablecoin_total_circulating_usd" in cols

    def test_config(self) -> None:
        strategy = MacroLiqTrendStrategy()
        assert isinstance(strategy.config, MacroLiqTrendConfig)

    def test_preprocess(self, sample_macro_df: pd.DataFrame) -> None:
        strategy = MacroLiqTrendStrategy()
        result = strategy.preprocess(sample_macro_df)
        assert len(result) == len(sample_macro_df)

    def test_generate_signals(self, sample_macro_df: pd.DataFrame) -> None:
        strategy = MacroLiqTrendStrategy()
        df = strategy.preprocess(sample_macro_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_macro_df: pd.DataFrame) -> None:
        strategy = MacroLiqTrendStrategy()
        processed, signals = strategy.run(sample_macro_df)
        assert len(processed) == len(sample_macro_df)
        assert len(signals.entries) == len(sample_macro_df)

    def test_from_params(self) -> None:
        strategy = MacroLiqTrendStrategy.from_params(dxy_roc_period=30)
        assert isinstance(strategy, MacroLiqTrendStrategy)

    def test_recommended_config(self) -> None:
        config = MacroLiqTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = MacroLiqTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "dxy_roc_period" in info
        assert "zscore_window" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = MacroLiqTrendConfig(dxy_roc_period=30, zscore_window=60)
        strategy = MacroLiqTrendStrategy(config=config)
        assert strategy._config.dxy_roc_period == 30

    def test_params_property(self) -> None:
        strategy = MacroLiqTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "dxy_roc_period" in params
        assert "zscore_window" in params

    def test_repr(self) -> None:
        strategy = MacroLiqTrendStrategy()
        assert "macro-liq-trend" in strategy.name
        assert repr(strategy)  # truthy (not empty)


class TestEdgeCases:
    def test_all_nan_macro_rejected(self) -> None:
        """All-NaN macro column should be rejected by validate_input."""
        np.random.seed(42)
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = 50000 + np.cumsum(np.random.randn(n) * 500)
        high = close + np.abs(np.random.randn(n) * 300)
        low = close - np.abs(np.random.randn(n) * 300)
        open_ = close + np.random.randn(n) * 100
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
                "macro_dxy": np.nan,
                "macro_vix": 18.0,
                "macro_spy": 480.0,
                "oc_stablecoin_total_circulating_usd": 1.3e11,
            },
            index=dates,
        )
        strategy = MacroLiqTrendStrategy()
        with pytest.raises(ValueError, match="all NaN"):
            strategy.run(df)

    def test_warmup_periods(self) -> None:
        config = MacroLiqTrendConfig()
        warmup = config.warmup_periods()
        assert warmup >= config.zscore_window
