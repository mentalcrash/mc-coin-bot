"""Tests for Trend Factor Multi-Horizon strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.trend_factor_12h.config import TrendFactorConfig
from src.strategy.trend_factor_12h.strategy import TrendFactorStrategy


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "trend-factor-12h" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("trend-factor-12h")
        assert cls is TrendFactorStrategy


class TestTrendFactorStrategy:
    def test_name(self) -> None:
        strategy = TrendFactorStrategy()
        assert strategy.name == "trend-factor-12h"

    def test_required_columns(self) -> None:
        strategy = TrendFactorStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "open" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns

    def test_config(self) -> None:
        strategy = TrendFactorStrategy()
        assert isinstance(strategy.config, TrendFactorConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = TrendFactorStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = TrendFactorStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = TrendFactorStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = TrendFactorStrategy.from_params(horizon_1=3, horizon_2=8, horizon_3=15)
        assert isinstance(strategy, TrendFactorStrategy)

    def test_recommended_config(self) -> None:
        config = TrendFactorStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config
        assert "trailing_stop_enabled" in config

    def test_get_startup_info(self) -> None:
        strategy = TrendFactorStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "horizons" in info
        assert "entry_threshold" in info
        assert "vol_target" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = TrendFactorConfig(horizon_1=3, horizon_2=8, horizon_3=15)
        strategy = TrendFactorStrategy(config=config)
        assert strategy._config.horizon_1 == 3

    def test_params_property(self) -> None:
        strategy = TrendFactorStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "horizon_1" in params

    def test_repr(self) -> None:
        strategy = TrendFactorStrategy()
        assert "trend-factor-12h" in strategy.name
        assert repr(strategy)  # truthy (not empty)

    def test_default_config(self) -> None:
        """기본 config 없이 생성 가능."""
        strategy = TrendFactorStrategy(config=None)
        assert strategy._config.horizon_1 == 5
