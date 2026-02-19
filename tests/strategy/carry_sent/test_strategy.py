"""Tests for Carry-Sentiment Gate strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.carry_sent.config import CarrySentConfig
from src.strategy.carry_sent.strategy import CarrySentStrategy


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.uniform(-0.001, 0.001, n)
    fear_greed = np.random.randint(10, 90, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
            "oc_fear_greed": fear_greed,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "carry-sent" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("carry-sent")
        assert cls is CarrySentStrategy


class TestCarrySentStrategy:
    def test_name(self) -> None:
        strategy = CarrySentStrategy()
        assert strategy.name == "carry-sent"

    def test_required_columns(self) -> None:
        strategy = CarrySentStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "funding_rate" in strategy.required_columns
        assert "oc_fear_greed" in strategy.required_columns

    def test_config(self) -> None:
        strategy = CarrySentStrategy()
        assert isinstance(strategy.config, CarrySentConfig)

    def test_preprocess(self, sample_df: pd.DataFrame) -> None:
        strategy = CarrySentStrategy()
        result = strategy.preprocess(sample_df)
        assert len(result) == len(sample_df)

    def test_generate_signals(self, sample_df: pd.DataFrame) -> None:
        strategy = CarrySentStrategy()
        df = strategy.preprocess(sample_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_df: pd.DataFrame) -> None:
        strategy = CarrySentStrategy()
        processed, signals = strategy.run(sample_df)
        assert len(processed) == len(sample_df)
        assert len(signals.entries) == len(sample_df)

    def test_from_params(self) -> None:
        strategy = CarrySentStrategy.from_params(fr_lookback=5)
        assert isinstance(strategy, CarrySentStrategy)

    def test_recommended_config(self) -> None:
        config = CarrySentStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = CarrySentStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_custom_config(self) -> None:
        config = CarrySentConfig(fr_lookback=5)
        strategy = CarrySentStrategy(config=config)
        assert strategy._config.fr_lookback == 5

    def test_params_property(self) -> None:
        strategy = CarrySentStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "fr_lookback" in params

    def test_repr(self) -> None:
        strategy = CarrySentStrategy()
        assert "carry-sent" in strategy.name
        assert repr(strategy)
