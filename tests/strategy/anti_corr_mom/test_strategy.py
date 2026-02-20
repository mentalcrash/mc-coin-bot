"""Tests for Anti-Correlation Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.anti_corr_mom.config import AntiCorrMomConfig
from src.strategy.anti_corr_mom.strategy import AntiCorrMomStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "anti-corr-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("anti-corr-mom")
        assert cls is AntiCorrMomStrategy


class TestAntiCorrMomStrategy:
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
        assert AntiCorrMomStrategy().name == "anti-corr-mom"

    def test_required_columns(self) -> None:
        assert "close" in AntiCorrMomStrategy().required_columns

    def test_config(self) -> None:
        assert isinstance(AntiCorrMomStrategy().config, AntiCorrMomConfig)

    def test_run_pipeline(self) -> None:
        strategy = AntiCorrMomStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = AntiCorrMomStrategy.from_params(corr_window=50)
        assert isinstance(strategy, AntiCorrMomStrategy)

    def test_recommended_config(self) -> None:
        assert isinstance(AntiCorrMomStrategy.recommended_config(), dict)

    def test_get_startup_info(self) -> None:
        info = AntiCorrMomStrategy().get_startup_info()
        assert "corr_window" in info

    def test_params_property(self) -> None:
        assert "corr_window" in AntiCorrMomStrategy().params

    def test_repr(self) -> None:
        assert repr(AntiCorrMomStrategy())
