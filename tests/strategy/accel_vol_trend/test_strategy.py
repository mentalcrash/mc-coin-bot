"""Tests for Acceleration-Volatility Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.accel_vol_trend.config import AccelVolTrendConfig
from src.strategy.accel_vol_trend.strategy import AccelVolTrendStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "accel-vol-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("accel-vol-trend")
        assert cls is AccelVolTrendStrategy


class TestAccelVolTrendStrategy:
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
        assert AccelVolTrendStrategy().name == "accel-vol-trend"

    def test_required_columns(self) -> None:
        cols = AccelVolTrendStrategy().required_columns
        assert "close" in cols
        assert "volume" in cols

    def test_config(self) -> None:
        assert isinstance(AccelVolTrendStrategy().config, AccelVolTrendConfig)

    def test_run_pipeline(self) -> None:
        strategy = AccelVolTrendStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = AccelVolTrendStrategy.from_params(accel_fast=3)
        assert isinstance(strategy, AccelVolTrendStrategy)

    def test_recommended_config(self) -> None:
        config = AccelVolTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = AccelVolTrendStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        assert "accel_fast" in AccelVolTrendStrategy().params

    def test_repr(self) -> None:
        assert repr(AccelVolTrendStrategy())

    def test_custom_config(self) -> None:
        config = AccelVolTrendConfig(accel_fast=3, accel_slow=30)
        strategy = AccelVolTrendStrategy(config=config)
        assert strategy._config.accel_fast == 3
