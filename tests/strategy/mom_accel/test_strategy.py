"""Tests for Momentum Acceleration strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.mom_accel.config import MomAccelConfig
from src.strategy.mom_accel.strategy import MomAccelStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "mom-accel" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("mom-accel")
        assert cls is MomAccelStrategy


class TestMomAccelStrategy:
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
        assert MomAccelStrategy().name == "mom-accel"

    def test_required_columns(self) -> None:
        assert "close" in MomAccelStrategy().required_columns

    def test_config(self) -> None:
        assert isinstance(MomAccelStrategy().config, MomAccelConfig)

    def test_run_pipeline(self) -> None:
        strategy = MomAccelStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = MomAccelStrategy.from_params(fast_roc=5, slow_roc=20)
        assert isinstance(strategy, MomAccelStrategy)

    def test_recommended_config(self) -> None:
        assert isinstance(MomAccelStrategy.recommended_config(), dict)

    def test_get_startup_info(self) -> None:
        info = MomAccelStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        assert "fast_roc" in MomAccelStrategy().params

    def test_repr(self) -> None:
        assert repr(MomAccelStrategy())
