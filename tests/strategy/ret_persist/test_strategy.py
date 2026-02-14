"""Tests for Return Persistence Score strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.ret_persist.config import RetPersistConfig
from src.strategy.ret_persist.strategy import RetPersistStrategy


class TestRegistry:
    def test_registered(self) -> None:
        assert "ret-persist" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("ret-persist")
        assert cls is RetPersistStrategy


class TestRetPersistStrategy:
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
        assert RetPersistStrategy().name == "ret-persist"

    def test_config(self) -> None:
        assert isinstance(RetPersistStrategy().config, RetPersistConfig)

    def test_run_pipeline(self) -> None:
        processed, signals = RetPersistStrategy().run(self._make_df())
        assert len(processed) == 300
        assert len(signals.entries) == 300

    def test_from_params(self) -> None:
        strategy = RetPersistStrategy.from_params(persist_window=20)
        assert isinstance(strategy, RetPersistStrategy)

    def test_recommended_config(self) -> None:
        assert "stop_loss_pct" in RetPersistStrategy.recommended_config()

    def test_get_startup_info(self) -> None:
        assert len(RetPersistStrategy().get_startup_info()) > 0

    def test_params_property(self) -> None:
        assert "persist_window" in RetPersistStrategy().params

    def test_repr(self) -> None:
        assert repr(RetPersistStrategy())
