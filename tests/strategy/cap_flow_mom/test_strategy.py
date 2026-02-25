"""Tests for Capital Flow Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.cap_flow_mom.config import CapFlowMomConfig
from src.strategy.cap_flow_mom.strategy import CapFlowMomStrategy


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
        assert "cap-flow-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("cap-flow-mom")
        assert cls is CapFlowMomStrategy


class TestCapFlowMomStrategy:
    def test_name(self) -> None:
        strategy = CapFlowMomStrategy()
        assert strategy.name == "cap-flow-mom"

    def test_required_columns(self) -> None:
        strategy = CapFlowMomStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = CapFlowMomStrategy()
        assert isinstance(strategy.config, CapFlowMomConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CapFlowMomStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CapFlowMomStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CapFlowMomStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = CapFlowMomStrategy.from_params(fast_roc_period=10, slow_roc_period=40)
        assert isinstance(strategy, CapFlowMomStrategy)

    def test_recommended_config(self) -> None:
        config = CapFlowMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = CapFlowMomStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "fast_roc_period" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = CapFlowMomConfig(fast_roc_period=10, slow_roc_period=40)
        strategy = CapFlowMomStrategy(config=config)
        assert strategy._config.fast_roc_period == 10

    def test_params_property(self) -> None:
        strategy = CapFlowMomStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "fast_roc_period" in params

    def test_repr(self) -> None:
        strategy = CapFlowMomStrategy()
        assert "cap-flow-mom" in strategy.name
        assert repr(strategy)  # truthy (not empty)
