"""Tests for Adaptive ROC Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.aroc_mom.config import ArocMomConfig
from src.strategy.aroc_mom.strategy import ArocMomStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "aroc-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("aroc-mom")
        assert cls is ArocMomStrategy


class TestArocMomStrategy:
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
        strategy = ArocMomStrategy()
        assert strategy.name == "aroc-mom"

    def test_required_columns(self) -> None:
        strategy = ArocMomStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = ArocMomStrategy()
        assert isinstance(strategy.config, ArocMomConfig)

    def test_preprocess(self) -> None:
        strategy = ArocMomStrategy()
        result = strategy.preprocess(self._make_df())
        assert len(result) == 300

    def test_generate_signals(self) -> None:
        strategy = ArocMomStrategy()
        df = strategy.preprocess(self._make_df())
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self) -> None:
        strategy = ArocMomStrategy()
        processed, signals = strategy.run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = ArocMomStrategy.from_params(fast_lookback=5)
        assert isinstance(strategy, ArocMomStrategy)

    def test_recommended_config(self) -> None:
        config = ArocMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = ArocMomStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_params_property(self) -> None:
        strategy = ArocMomStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "fast_lookback" in params

    def test_repr(self) -> None:
        strategy = ArocMomStrategy()
        assert "aroc-mom" in strategy.name
        assert repr(strategy)
