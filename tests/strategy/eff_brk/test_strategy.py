"""Tests for Efficiency Breakout strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.eff_brk.config import EffBrkConfig
from src.strategy.eff_brk.strategy import EffBrkStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "eff-brk" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("eff-brk")
        assert cls is EffBrkStrategy


class TestEffBrkStrategy:
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
        assert EffBrkStrategy().name == "eff-brk"

    def test_required_columns(self) -> None:
        assert "close" in EffBrkStrategy().required_columns

    def test_config(self) -> None:
        assert isinstance(EffBrkStrategy().config, EffBrkConfig)

    def test_run_pipeline(self) -> None:
        processed, signals = EffBrkStrategy().run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = EffBrkStrategy.from_params(er_period=20)
        assert isinstance(strategy, EffBrkStrategy)

    def test_recommended_config(self) -> None:
        config = EffBrkStrategy.recommended_config()
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = EffBrkStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        params = EffBrkStrategy().params
        assert "er_period" in params

    def test_repr(self) -> None:
        assert repr(EffBrkStrategy())
