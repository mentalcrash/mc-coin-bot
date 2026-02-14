"""Tests for OBV Acceleration Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.obv_accel.config import ObvAccelConfig
from src.strategy.obv_accel.strategy import ObvAccelStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "obv-accel" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("obv-accel")
        assert cls is ObvAccelStrategy


class TestObvAccelStrategy:
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
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

    def test_name(self) -> None:
        assert ObvAccelStrategy().name == "obv-accel"

    def test_config(self) -> None:
        assert isinstance(ObvAccelStrategy().config, ObvAccelConfig)

    def test_run_pipeline(self) -> None:
        processed, signals = ObvAccelStrategy().run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = ObvAccelStrategy.from_params(obv_smooth=15)
        assert isinstance(strategy, ObvAccelStrategy)

    def test_recommended_config(self) -> None:
        assert "stop_loss_pct" in ObvAccelStrategy.recommended_config()

    def test_get_startup_info(self) -> None:
        assert len(ObvAccelStrategy().get_startup_info()) > 0

    def test_params_property(self) -> None:
        assert "obv_smooth" in ObvAccelStrategy().params

    def test_repr(self) -> None:
        assert repr(ObvAccelStrategy())
