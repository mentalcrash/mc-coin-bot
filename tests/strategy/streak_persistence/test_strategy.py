"""Tests for Return Streak Persistence strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.streak_persistence.config import StreakPersistenceConfig
from src.strategy.streak_persistence.strategy import StreakPersistenceStrategy


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
        assert "streak-persistence" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("streak-persistence")
        assert cls is StreakPersistenceStrategy


class TestStreakPersistenceStrategy:
    def test_name(self) -> None:
        strategy = StreakPersistenceStrategy()
        assert strategy.name == "streak-persistence"

    def test_required_columns(self) -> None:
        strategy = StreakPersistenceStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = StreakPersistenceStrategy()
        assert isinstance(strategy.config, StreakPersistenceConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = StreakPersistenceStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = StreakPersistenceStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = StreakPersistenceStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = StreakPersistenceStrategy.from_params(streak_threshold=4)
        assert isinstance(strategy, StreakPersistenceStrategy)

    def test_recommended_config(self) -> None:
        config = StreakPersistenceStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = StreakPersistenceStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "streak_threshold" in info

    def test_custom_config(self) -> None:
        config = StreakPersistenceConfig(streak_threshold=4)
        strategy = StreakPersistenceStrategy(config=config)
        assert strategy._config.streak_threshold == 4

    def test_params_property(self) -> None:
        strategy = StreakPersistenceStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "streak_threshold" in params

    def test_repr(self) -> None:
        strategy = StreakPersistenceStrategy()
        assert "streak-persistence" in strategy.name
        assert repr(strategy)
