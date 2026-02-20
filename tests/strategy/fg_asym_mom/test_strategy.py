"""Tests for FgAsymMom strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.fg_asym_mom.config import FgAsymMomConfig
from src.strategy.fg_asym_mom.strategy import FgAsymMomStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "fg-asym-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("fg-asym-mom")
        assert cls is FgAsymMomStrategy


class TestFgAsymMomStrategy:
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
        assert FgAsymMomStrategy().name == "fg-asym-mom"

    def test_required_columns(self) -> None:
        assert "close" in FgAsymMomStrategy().required_columns
        assert "oc_fear_greed" in FgAsymMomStrategy().required_columns

    def test_config(self) -> None:
        assert isinstance(FgAsymMomStrategy().config, FgAsymMomConfig)

    def test_run_pipeline(self) -> None:
        strategy = FgAsymMomStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = FgAsymMomStrategy.from_params(fear_threshold=20.0)
        assert isinstance(strategy, FgAsymMomStrategy)

    def test_recommended_config(self) -> None:
        assert isinstance(FgAsymMomStrategy.recommended_config(), dict)

    def test_get_startup_info(self) -> None:
        info = FgAsymMomStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        assert "fear_threshold" in FgAsymMomStrategy().params

    def test_repr(self) -> None:
        assert repr(FgAsymMomStrategy())
