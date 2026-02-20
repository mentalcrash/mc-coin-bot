"""Tests for VRP-Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.vrp_trend.config import VrpTrendConfig
from src.strategy.vrp_trend.strategy import VrpTrendStrategy


@pytest.fixture
def sample_ohlcv_dvol_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    dvol = 50.0 + np.cumsum(np.random.randn(n) * 2)
    dvol = np.clip(dvol, 20, 120)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "dvol": dvol,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "vrp-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("vrp-trend")
        assert cls is VrpTrendStrategy


class TestVrpTrendStrategy:
    def test_name(self) -> None:
        strategy = VrpTrendStrategy()
        assert strategy.name == "vrp-trend"

    def test_required_columns(self) -> None:
        strategy = VrpTrendStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "dvol" in strategy.required_columns

    def test_config(self) -> None:
        strategy = VrpTrendStrategy()
        assert isinstance(strategy.config, VrpTrendConfig)

    def test_preprocess(self, sample_ohlcv_dvol_df: pd.DataFrame) -> None:
        strategy = VrpTrendStrategy()
        result = strategy.preprocess(sample_ohlcv_dvol_df)
        assert len(result) == len(sample_ohlcv_dvol_df)

    def test_generate_signals(self, sample_ohlcv_dvol_df: pd.DataFrame) -> None:
        strategy = VrpTrendStrategy()
        df = strategy.preprocess(sample_ohlcv_dvol_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_dvol_df: pd.DataFrame) -> None:
        strategy = VrpTrendStrategy()
        processed, signals = strategy.run(sample_ohlcv_dvol_df)
        assert len(processed) == len(sample_ohlcv_dvol_df)
        assert len(signals.entries) == len(sample_ohlcv_dvol_df)

    def test_from_params(self) -> None:
        strategy = VrpTrendStrategy.from_params(rv_window=20)
        assert isinstance(strategy, VrpTrendStrategy)

    def test_recommended_config(self) -> None:
        config = VrpTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = VrpTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "rv_window" in info
        assert "vrp_entry_z" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = VrpTrendConfig(rv_window=20)
        strategy = VrpTrendStrategy(config=config)
        assert strategy._config.rv_window == 20

    def test_params_property(self) -> None:
        strategy = VrpTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "rv_window" in params
        assert "vrp_entry_z" in params

    def test_repr(self) -> None:
        strategy = VrpTrendStrategy()
        assert "vrp-trend" in strategy.name
        assert repr(strategy)  # truthy (not empty)
