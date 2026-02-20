"""Tests for GBTrend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.gbtrend.config import GBTrendConfig
from src.strategy.gbtrend.strategy import GBTrendStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "gbtrend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("gbtrend")
        assert cls is GBTrendStrategy


class TestGBTrendStrategy:
    def test_name(self) -> None:
        strategy = GBTrendStrategy()
        assert strategy.name == "gbtrend"

    def test_required_columns(self) -> None:
        strategy = GBTrendStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = GBTrendStrategy()
        assert isinstance(strategy.config, GBTrendConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = GBTrendStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = GBTrendStrategy(
            config=GBTrendConfig(training_window=60, n_estimators=10, max_depth=2)
        )
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = GBTrendStrategy(
            config=GBTrendConfig(training_window=60, n_estimators=10, max_depth=2)
        )
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = GBTrendStrategy.from_params(training_window=120)
        assert isinstance(strategy, GBTrendStrategy)

    def test_recommended_config(self) -> None:
        config = GBTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = GBTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "n_estimators" in info

    def test_custom_config(self) -> None:
        config = GBTrendConfig(n_estimators=50)
        strategy = GBTrendStrategy(config=config)
        assert strategy._config.n_estimators == 50  # noqa: SLF001

    def test_params_property(self) -> None:
        strategy = GBTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "training_window" in params

    def test_repr(self) -> None:
        strategy = GBTrendStrategy()
        assert "gbtrend" in strategy.name
        assert repr(strategy)
