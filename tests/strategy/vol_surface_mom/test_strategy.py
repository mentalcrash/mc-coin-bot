"""Tests for Volatility Surface Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.vol_surface_mom.config import VolSurfaceMomConfig
from src.strategy.vol_surface_mom.strategy import VolSurfaceMomStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "vol-surface-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("vol-surface-mom")
        assert cls is VolSurfaceMomStrategy


class TestVolSurfaceMomStrategy:
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
        assert VolSurfaceMomStrategy().name == "vol-surface-mom"

    def test_required_columns(self) -> None:
        cols = VolSurfaceMomStrategy().required_columns
        assert "close" in cols
        assert "volume" in cols

    def test_config(self) -> None:
        assert isinstance(VolSurfaceMomStrategy().config, VolSurfaceMomConfig)

    def test_run_pipeline(self) -> None:
        strategy = VolSurfaceMomStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = VolSurfaceMomStrategy.from_params(gk_window=30)
        assert isinstance(strategy, VolSurfaceMomStrategy)

    def test_recommended_config(self) -> None:
        config = VolSurfaceMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = VolSurfaceMomStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        assert "gk_window" in VolSurfaceMomStrategy().params

    def test_repr(self) -> None:
        assert repr(VolSurfaceMomStrategy())

    def test_custom_config(self) -> None:
        config = VolSurfaceMomConfig(gk_window=30)
        strategy = VolSurfaceMomStrategy(config=config)
        assert strategy._config.gk_window == 30
