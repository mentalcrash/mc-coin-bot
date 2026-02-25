"""Tests for ER Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.er_trend.config import ErTrendConfig
from src.strategy.er_trend.strategy import ErTrendStrategy


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
        assert "er-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("er-trend")
        assert cls is ErTrendStrategy


class TestErTrendStrategy:
    def test_name(self) -> None:
        strategy = ErTrendStrategy()
        assert strategy.name == "er-trend"

    def test_required_columns(self) -> None:
        strategy = ErTrendStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = ErTrendStrategy()
        assert isinstance(strategy.config, ErTrendConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = ErTrendStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = ErTrendStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = ErTrendStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = ErTrendStrategy.from_params(er_fast=5, er_mid=15, er_slow=30)
        assert isinstance(strategy, ErTrendStrategy)

    def test_recommended_config(self) -> None:
        config = ErTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = ErTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "er_fast" in info
        assert "er_mid" in info
        assert "er_slow" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = ErTrendConfig(er_fast=5, er_mid=15, er_slow=30)
        strategy = ErTrendStrategy(config=config)
        assert strategy._config.er_fast == 5

    def test_params_property(self) -> None:
        strategy = ErTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "er_fast" in params

    def test_repr(self) -> None:
        strategy = ErTrendStrategy()
        assert "er-trend" in strategy.name
        assert repr(strategy)  # truthy (not empty)

    def test_warmup_periods_accessible(self) -> None:
        strategy = ErTrendStrategy()
        config: ErTrendConfig = strategy._config
        assert config.warmup_periods() > 0

    def test_direction_range(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """앙상블 호환성: direction 값 범위 {-1, 0, 1}."""
        strategy = ErTrendStrategy()
        _, signals = strategy.run(sample_ohlcv_df)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})
