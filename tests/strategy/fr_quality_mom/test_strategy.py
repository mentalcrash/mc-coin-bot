"""Tests for FR Quality Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.fr_quality_mom.config import FrQualityMomConfig
from src.strategy.fr_quality_mom.strategy import FrQualityMomStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "fr-quality-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("fr-quality-mom")
        assert cls is FrQualityMomStrategy


class TestFrQualityMomStrategy:
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
        funding_rate = np.random.uniform(-0.001, 0.001, n)
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "funding_rate": funding_rate,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1D"),
        )

    def test_name(self) -> None:
        assert FrQualityMomStrategy().name == "fr-quality-mom"

    def test_required_columns(self) -> None:
        cols = FrQualityMomStrategy().required_columns
        assert "close" in cols
        assert "funding_rate" in cols

    def test_config(self) -> None:
        assert isinstance(FrQualityMomStrategy().config, FrQualityMomConfig)

    def test_run_pipeline(self) -> None:
        strategy = FrQualityMomStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = FrQualityMomStrategy.from_params(fr_lookback=14)
        assert isinstance(strategy, FrQualityMomStrategy)

    def test_recommended_config(self) -> None:
        assert isinstance(FrQualityMomStrategy.recommended_config(), dict)

    def test_get_startup_info(self) -> None:
        info = FrQualityMomStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        assert "momentum_window" in FrQualityMomStrategy().params

    def test_repr(self) -> None:
        assert repr(FrQualityMomStrategy())
