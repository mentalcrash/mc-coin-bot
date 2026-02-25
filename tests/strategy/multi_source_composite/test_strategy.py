"""Tests for Multi-Source Directional Composite strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.multi_source_composite.config import MultiSourceCompositeConfig
from src.strategy.multi_source_composite.strategy import MultiSourceCompositeStrategy


@property
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
    fg = np.clip(50 + np.cumsum(np.random.randn(n) * 3), 0, 100).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_fear_greed": fg,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


def _make_sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    fg = np.clip(50 + np.cumsum(np.random.randn(n) * 3), 0, 100).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_fear_greed": fg,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "multi-source-composite" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("multi-source-composite")
        assert cls is MultiSourceCompositeStrategy


class TestMultiSourceCompositeStrategy:
    def test_name(self) -> None:
        strategy = MultiSourceCompositeStrategy()
        assert strategy.name == "multi-source-composite"

    def test_required_columns(self) -> None:
        strategy = MultiSourceCompositeStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "oc_fear_greed" in strategy.required_columns

    def test_config(self) -> None:
        strategy = MultiSourceCompositeStrategy()
        assert isinstance(strategy.config, MultiSourceCompositeConfig)

    def test_preprocess(self) -> None:
        strategy = MultiSourceCompositeStrategy()
        df = _make_sample_df()
        result = strategy.preprocess(df)
        assert len(result) == len(df)

    def test_generate_signals(self) -> None:
        strategy = MultiSourceCompositeStrategy()
        df = _make_sample_df()
        processed = strategy.preprocess(df)
        signals = strategy.generate_signals(processed)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self) -> None:
        strategy = MultiSourceCompositeStrategy()
        df = _make_sample_df()
        processed, signals = strategy.run(df)
        assert len(processed) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self) -> None:
        strategy = MultiSourceCompositeStrategy.from_params(mom_fast=5, mom_slow=50)
        assert isinstance(strategy, MultiSourceCompositeStrategy)

    def test_recommended_config(self) -> None:
        config = MultiSourceCompositeStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = MultiSourceCompositeStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "mom_fast" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = MultiSourceCompositeConfig(mom_fast=5, mom_slow=50)
        strategy = MultiSourceCompositeStrategy(config=config)
        assert strategy._config.mom_fast == 5

    def test_params_property(self) -> None:
        strategy = MultiSourceCompositeStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "mom_fast" in params
        assert "velocity_fast_window" in params
        assert "fg_delta_window" in params

    def test_repr(self) -> None:
        strategy = MultiSourceCompositeStrategy()
        assert "multi-source-composite" in strategy.name
        assert repr(strategy)  # truthy (not empty)
