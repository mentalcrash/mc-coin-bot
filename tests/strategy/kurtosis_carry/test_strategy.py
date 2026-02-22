"""Tests for Kurtosis Carry strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.kurtosis_carry.config import KurtosisCarryConfig
from src.strategy.kurtosis_carry.strategy import KurtosisCarryStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "kurtosis-carry" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("kurtosis-carry")
        assert cls is KurtosisCarryStrategy


class TestKurtosisCarryStrategy:
    def test_name(self) -> None:
        strategy = KurtosisCarryStrategy()
        assert strategy.name == "kurtosis-carry"

    def test_required_columns(self) -> None:
        strategy = KurtosisCarryStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = KurtosisCarryStrategy()
        assert isinstance(strategy.config, KurtosisCarryConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = KurtosisCarryStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = KurtosisCarryStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = KurtosisCarryStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = KurtosisCarryStrategy.from_params(kurtosis_window=20, kurtosis_long_window=60)
        assert isinstance(strategy, KurtosisCarryStrategy)

    def test_recommended_config(self) -> None:
        config = KurtosisCarryStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = KurtosisCarryStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "kurtosis_window" in info

    def test_custom_config(self) -> None:
        config = KurtosisCarryConfig(kurtosis_window=20, kurtosis_long_window=60)
        strategy = KurtosisCarryStrategy(config=config)
        assert strategy._config.kurtosis_window == 20

    def test_params_property(self) -> None:
        strategy = KurtosisCarryStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "kurtosis_window" in params

    def test_repr(self) -> None:
        strategy = KurtosisCarryStrategy()
        assert "kurtosis-carry" in strategy.name
        assert repr(strategy)
