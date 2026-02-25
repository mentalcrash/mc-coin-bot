"""Tests for Trend Quality Momentum (TQ-Mom) strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.tq_mom.config import TqMomConfig
from src.strategy.tq_mom.strategy import TqMomStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "tq-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("tq-mom")
        assert cls is TqMomStrategy


class TestTqMomStrategy:
    def test_name(self) -> None:
        strategy = TqMomStrategy()
        assert strategy.name == "tq-mom"

    def test_required_columns(self) -> None:
        strategy = TqMomStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = TqMomStrategy()
        assert isinstance(strategy.config, TqMomConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = TqMomStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = TqMomStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = TqMomStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = TqMomStrategy.from_params(hurst_window=60)
        assert isinstance(strategy, TqMomStrategy)

    def test_recommended_config(self) -> None:
        config = TqMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = TqMomStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "hurst_window" in info
        assert "fd_period" in info

    def test_custom_config(self) -> None:
        config = TqMomConfig(hurst_window=60)
        strategy = TqMomStrategy(config=config)
        assert strategy._config.hurst_window == 60

    def test_params_property(self) -> None:
        strategy = TqMomStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "hurst_window" in params

    def test_repr(self) -> None:
        strategy = TqMomStrategy()
        assert "tq-mom" in strategy.name
        assert repr(strategy)
