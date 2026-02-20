"""Tests for Return Distribution Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.dist_mom.config import DistMomConfig
from src.strategy.dist_mom.strategy import DistMomStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "dist-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("dist-mom")
        assert cls is DistMomStrategy


class TestDistMomStrategy:
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
        assert DistMomStrategy().name == "dist-mom"

    def test_required_columns(self) -> None:
        assert "close" in DistMomStrategy().required_columns

    def test_config(self) -> None:
        assert isinstance(DistMomStrategy().config, DistMomConfig)

    def test_run_pipeline(self) -> None:
        strategy = DistMomStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = DistMomStrategy.from_params(dist_window=30)
        assert isinstance(strategy, DistMomStrategy)

    def test_recommended_config(self) -> None:
        assert isinstance(DistMomStrategy.recommended_config(), dict)

    def test_get_startup_info(self) -> None:
        info = DistMomStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        assert "dist_window" in DistMomStrategy().params

    def test_repr(self) -> None:
        assert repr(DistMomStrategy())
