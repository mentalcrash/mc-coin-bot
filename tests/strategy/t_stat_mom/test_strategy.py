"""Tests for T-Stat Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.t_stat_mom.config import TStatMomConfig
from src.strategy.t_stat_mom.strategy import TStatMomStrategy


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
        assert "t-stat-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("t-stat-mom")
        assert cls is TStatMomStrategy


class TestTStatMomStrategy:
    def test_name(self) -> None:
        strategy = TStatMomStrategy()
        assert strategy.name == "t-stat-mom"

    def test_required_columns(self) -> None:
        strategy = TStatMomStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "open" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns

    def test_config(self) -> None:
        strategy = TStatMomStrategy()
        assert isinstance(strategy.config, TStatMomConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = TStatMomStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = TStatMomStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = TStatMomStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = TStatMomStrategy.from_params(fast_lookback=10, mid_lookback=30, slow_lookback=60)
        assert isinstance(strategy, TStatMomStrategy)

    def test_recommended_config(self) -> None:
        config = TStatMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config
        assert "trailing_stop_enabled" in config

    def test_get_startup_info(self) -> None:
        strategy = TStatMomStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "fast_lookback" in info
        assert "mid_lookback" in info
        assert "slow_lookback" in info
        assert "entry_threshold" in info
        assert "tanh_scale" in info
        assert "vol_target" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = TStatMomConfig(fast_lookback=10, mid_lookback=30, slow_lookback=60)
        strategy = TStatMomStrategy(config=config)
        assert strategy._config.fast_lookback == 10

    def test_params_property(self) -> None:
        strategy = TStatMomStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "fast_lookback" in params

    def test_repr(self) -> None:
        strategy = TStatMomStrategy()
        assert "t-stat-mom" in strategy.name
        assert repr(strategy)
