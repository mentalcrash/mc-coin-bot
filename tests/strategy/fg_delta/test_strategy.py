"""Tests for FgDelta strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.fg_delta.config import FgDeltaConfig
from src.strategy.fg_delta.strategy import FgDeltaStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "fg-delta" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("fg-delta")
        assert cls is FgDeltaStrategy


class TestFgDeltaStrategy:
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
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="1D"),
        )
        df["oc_fear_greed"] = np.random.randint(10, 90, n).astype(float)
        return df

    def test_name(self) -> None:
        assert FgDeltaStrategy().name == "fg-delta"

    def test_required_columns(self) -> None:
        assert "close" in FgDeltaStrategy().required_columns

    def test_config(self) -> None:
        assert isinstance(FgDeltaStrategy().config, FgDeltaConfig)

    def test_run_pipeline(self) -> None:
        strategy = FgDeltaStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = FgDeltaStrategy.from_params(fg_delta_window=10)
        assert isinstance(strategy, FgDeltaStrategy)

    def test_recommended_config(self) -> None:
        assert isinstance(FgDeltaStrategy.recommended_config(), dict)

    def test_get_startup_info(self) -> None:
        info = FgDeltaStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        assert "fg_delta_window" in FgDeltaStrategy().params

    def test_repr(self) -> None:
        assert repr(FgDeltaStrategy())
