"""Tests for Weekend-Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.weekend_mom.config import WeekendMomConfig
from src.strategy.weekend_mom.strategy import WeekendMomStrategy


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
        assert "weekend-mom-12h" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("weekend-mom-12h")
        assert cls is WeekendMomStrategy


class TestWeekendMomStrategy:
    def test_name(self) -> None:
        strategy = WeekendMomStrategy()
        assert strategy.name == "weekend-mom-12h"

    def test_required_columns(self) -> None:
        strategy = WeekendMomStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = WeekendMomStrategy()
        assert isinstance(strategy.config, WeekendMomConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = WeekendMomStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = WeekendMomStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = WeekendMomStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = WeekendMomStrategy.from_params(fast_lookback=10, slow_lookback=40)
        assert isinstance(strategy, WeekendMomStrategy)

    def test_recommended_config(self) -> None:
        config = WeekendMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = WeekendMomStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "fast_lookback" in info
        assert "weekend_boost" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = WeekendMomConfig(fast_lookback=10, slow_lookback=40)
        strategy = WeekendMomStrategy(config=config)
        assert strategy._config.fast_lookback == 10

    def test_params_property(self) -> None:
        strategy = WeekendMomStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "fast_lookback" in params

    def test_repr(self) -> None:
        strategy = WeekendMomStrategy()
        assert "weekend-mom-12h" in strategy.name
        assert repr(strategy)  # truthy (not empty)

    @pytest.mark.parametrize("boost", [1.0, 2.0, 3.5])
    def test_various_boosts(self, sample_ohlcv_df: pd.DataFrame, boost: float) -> None:
        """Different weekend_boost values should all produce valid signals."""
        config = WeekendMomConfig(weekend_boost=boost)
        strategy = WeekendMomStrategy(config=config)
        _processed, signals = strategy.run(sample_ohlcv_df)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})
        assert not signals.strength.isna().any()
