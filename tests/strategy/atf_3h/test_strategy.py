"""Tests for atf-3h strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.atf_3h.config import Atf3hConfig
from src.strategy.atf_3h.strategy import Atf3hStrategy


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
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
        index=pd.date_range("2024-01-01", periods=n, freq="3h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "atf-3h" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("atf-3h")
        assert cls is Atf3hStrategy


class TestAtf3hStrategy:
    def test_name(self) -> None:
        strategy = Atf3hStrategy()
        assert strategy.name == "atf-3h"

    def test_required_columns(self) -> None:
        strategy = Atf3hStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = Atf3hStrategy()
        assert isinstance(strategy.config, Atf3hConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = Atf3hStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = Atf3hStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = Atf3hStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = Atf3hStrategy.from_params(nearness_lookback=30)
        assert isinstance(strategy, Atf3hStrategy)

    def test_recommended_config(self) -> None:
        config = Atf3hStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = Atf3hStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_custom_config(self) -> None:
        config = Atf3hConfig(nearness_lookback=30)
        strategy = Atf3hStrategy(config=config)
        assert strategy._config.nearness_lookback == 30

    def test_params_property(self) -> None:
        strategy = Atf3hStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "nearness_lookback" in params

    def test_repr(self) -> None:
        strategy = Atf3hStrategy()
        assert "atf-3h" in strategy.name
        assert repr(strategy)
