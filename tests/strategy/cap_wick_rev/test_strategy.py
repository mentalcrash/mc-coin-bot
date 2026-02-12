"""Tests for Capitulation Wick Reversal strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.cap_wick_rev.config import CapWickRevConfig
from src.strategy.cap_wick_rev.strategy import CapWickRevStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "cap-wick-rev" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("cap-wick-rev")
        assert cls is CapWickRevStrategy


class TestCapWickRevStrategy:
    def test_name(self) -> None:
        strategy = CapWickRevStrategy()
        assert strategy.name == "cap-wick-rev"

    def test_required_columns(self) -> None:
        strategy = CapWickRevStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "open" in strategy.required_columns

    def test_config(self) -> None:
        strategy = CapWickRevStrategy()
        assert isinstance(strategy.config, CapWickRevConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CapWickRevStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CapWickRevStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CapWickRevStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = CapWickRevStrategy.from_params(atr_spike_threshold=2.5)
        assert isinstance(strategy, CapWickRevStrategy)
        assert strategy._config.atr_spike_threshold == 2.5

    def test_recommended_config(self) -> None:
        config = CapWickRevStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "system_stop_loss" in config

    def test_get_startup_info(self) -> None:
        strategy = CapWickRevStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "atr_spike" in info
        assert "mode" in info

    def test_warmup_periods(self) -> None:
        strategy = CapWickRevStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup >= strategy._config.atr_window

    def test_custom_config(self) -> None:
        config = CapWickRevConfig(atr_spike_threshold=3.0)
        strategy = CapWickRevStrategy(config=config)
        assert strategy._config.atr_spike_threshold == 3.0

    def test_params_property(self) -> None:
        strategy = CapWickRevStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "atr_spike_threshold" in params

    def test_repr(self) -> None:
        strategy = CapWickRevStrategy()
        assert "cap-wick-rev" in strategy.name
        assert repr(strategy)  # truthy (not empty)
