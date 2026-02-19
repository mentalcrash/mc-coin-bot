"""Tests for Liquidity-Confirmed Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.liq_conf_trend.config import LiqConfTrendConfig
from src.strategy.liq_conf_trend.strategy import LiqConfTrendStrategy


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
    stablecoin = 100e9 + np.cumsum(np.random.randn(n) * 1e8)
    tvl = 50e9 + np.cumsum(np.random.randn(n) * 5e7)
    fear_greed = np.random.randint(10, 90, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_stablecoin_total_usd": stablecoin,
            "oc_tvl_usd": tvl,
            "oc_fear_greed": fear_greed,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def sample_ohlcv_only_df() -> pd.DataFrame:
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
        assert "liq-conf-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("liq-conf-trend")
        assert cls is LiqConfTrendStrategy


class TestLiqConfTrendStrategy:
    def test_name(self) -> None:
        strategy = LiqConfTrendStrategy()
        assert strategy.name == "liq-conf-trend"

    def test_required_columns(self) -> None:
        strategy = LiqConfTrendStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = LiqConfTrendStrategy()
        assert isinstance(strategy.config, LiqConfTrendConfig)

    def test_preprocess(self, sample_df: pd.DataFrame) -> None:
        strategy = LiqConfTrendStrategy()
        result = strategy.preprocess(sample_df)
        assert len(result) == len(sample_df)

    def test_generate_signals(self, sample_df: pd.DataFrame) -> None:
        strategy = LiqConfTrendStrategy()
        df = strategy.preprocess(sample_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_df: pd.DataFrame) -> None:
        strategy = LiqConfTrendStrategy()
        processed, signals = strategy.run(sample_df)
        assert len(processed) == len(sample_df)
        assert len(signals.entries) == len(sample_df)

    def test_run_pipeline_ohlcv_only(self, sample_ohlcv_only_df: pd.DataFrame) -> None:
        """Graceful degradation with OHLCV only."""
        strategy = LiqConfTrendStrategy()
        processed, signals = strategy.run(sample_ohlcv_only_df)
        assert len(processed) == len(sample_ohlcv_only_df)
        assert len(signals.entries) == len(sample_ohlcv_only_df)

    def test_from_params(self) -> None:
        strategy = LiqConfTrendStrategy.from_params(mom_lookback=30)
        assert isinstance(strategy, LiqConfTrendStrategy)

    def test_recommended_config(self) -> None:
        config = LiqConfTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = LiqConfTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_custom_config(self) -> None:
        config = LiqConfTrendConfig(mom_lookback=30)
        strategy = LiqConfTrendStrategy(config=config)
        assert strategy._config.mom_lookback == 30

    def test_params_property(self) -> None:
        strategy = LiqConfTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "mom_lookback" in params

    def test_repr(self) -> None:
        strategy = LiqConfTrendStrategy()
        assert "liq-conf-trend" in strategy.name
        assert repr(strategy)
