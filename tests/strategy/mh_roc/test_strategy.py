"""Tests for Multi-Horizon ROC Ensemble strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.mh_roc.config import MhRocConfig
from src.strategy.mh_roc.strategy import MhRocStrategy


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
        assert "mh-roc" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("mh-roc")
        assert cls is MhRocStrategy


class TestMhRocStrategy:
    def test_name(self) -> None:
        strategy = MhRocStrategy()
        assert strategy.name == "mh-roc"

    def test_required_columns(self) -> None:
        strategy = MhRocStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = MhRocStrategy()
        assert isinstance(strategy.config, MhRocConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = MhRocStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = MhRocStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = MhRocStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = MhRocStrategy.from_params(roc_short=8, roc_medium_short=24)
        assert isinstance(strategy, MhRocStrategy)

    def test_recommended_config(self) -> None:
        config = MhRocStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = MhRocStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "roc_horizons" in info

    def test_custom_config(self) -> None:
        config = MhRocConfig(roc_short=8, roc_medium_short=24)
        strategy = MhRocStrategy(config=config)
        assert strategy._config.roc_short == 8

    def test_params_property(self) -> None:
        strategy = MhRocStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "roc_short" in params

    def test_repr(self) -> None:
        strategy = MhRocStrategy()
        assert "mh-roc" in strategy.name
        assert repr(strategy)  # truthy (not empty)
