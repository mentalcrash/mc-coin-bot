"""Tests for dir-vol-trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.dir_vol_trend.config import DirVolTrendConfig
from src.strategy.dir_vol_trend.strategy import DirVolTrendStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="6h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "dir-vol-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("dir-vol-trend")
        assert cls is DirVolTrendStrategy


class TestDirVolTrendStrategy:
    def test_name(self) -> None:
        strategy = DirVolTrendStrategy()
        assert strategy.name == "dir-vol-trend"

    def test_required_columns(self) -> None:
        strategy = DirVolTrendStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = DirVolTrendStrategy()
        assert isinstance(strategy.config, DirVolTrendConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = DirVolTrendStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = DirVolTrendStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = DirVolTrendStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = DirVolTrendStrategy.from_params(dvt_window=10)
        assert isinstance(strategy, DirVolTrendStrategy)

    def test_recommended_config(self) -> None:
        config = DirVolTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = DirVolTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_custom_config(self) -> None:
        config = DirVolTrendConfig(dvt_window=10)
        strategy = DirVolTrendStrategy(config=config)
        assert strategy._config.dvt_window == 10

    def test_params_property(self) -> None:
        strategy = DirVolTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "dvt_window" in params

    def test_repr(self) -> None:
        strategy = DirVolTrendStrategy()
        assert "dir-vol-trend" in strategy.name
        assert repr(strategy)
