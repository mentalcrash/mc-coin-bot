"""Tests for Complexity-Filtered Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.complexity_trend.config import ComplexityTrendConfig
from src.strategy.complexity_trend.strategy import ComplexityTrendStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "complexity-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("complexity-trend")
        assert cls is ComplexityTrendStrategy


class TestComplexityTrendStrategy:
    def _make_df(self) -> pd.DataFrame:
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
            index=pd.date_range("2024-01-01", periods=n, freq="1D"),
        )

    def test_name(self) -> None:
        assert ComplexityTrendStrategy().name == "complexity-trend"

    def test_required_columns(self) -> None:
        cols = ComplexityTrendStrategy().required_columns
        assert "close" in cols
        assert "volume" in cols

    def test_config(self) -> None:
        assert isinstance(ComplexityTrendStrategy().config, ComplexityTrendConfig)

    def test_run_pipeline(self) -> None:
        strategy = ComplexityTrendStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = ComplexityTrendStrategy.from_params(hurst_window=80)
        assert isinstance(strategy, ComplexityTrendStrategy)

    def test_recommended_config(self) -> None:
        config = ComplexityTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = ComplexityTrendStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        assert "hurst_window" in ComplexityTrendStrategy().params

    def test_repr(self) -> None:
        assert repr(ComplexityTrendStrategy())

    def test_custom_config(self) -> None:
        config = ComplexityTrendConfig(hurst_window=80)
        strategy = ComplexityTrendStrategy(config=config)
        assert strategy._config.hurst_window == 80
