"""Tests for Z-Momentum (MACD-V) strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.z_mom.config import ZMomConfig
from src.strategy.z_mom.strategy import ZMomStrategy


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
        assert "z-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("z-mom")
        assert cls is ZMomStrategy


class TestZMomStrategy:
    def test_name(self) -> None:
        strategy = ZMomStrategy()
        assert strategy.name == "z-mom"

    def test_required_columns(self) -> None:
        strategy = ZMomStrategy()
        assert "close" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "open" in strategy.required_columns

    def test_config(self) -> None:
        strategy = ZMomStrategy()
        assert isinstance(strategy.config, ZMomConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = ZMomStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)
        assert "macd_v" in result.columns

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = ZMomStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = ZMomStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = ZMomStrategy.from_params(macd_fast=8, macd_slow=21)
        assert isinstance(strategy, ZMomStrategy)

    def test_from_params_custom_flat_zone(self) -> None:
        strategy = ZMomStrategy.from_params(flat_zone=1.5)
        assert isinstance(strategy, ZMomStrategy)
        assert strategy._config.flat_zone == 1.5

    def test_recommended_config(self) -> None:
        config = ZMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config
        assert "trailing_stop_enabled" in config

    def test_get_startup_info(self) -> None:
        strategy = ZMomStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "macd_fast" in info
        assert "macd_slow" in info
        assert "flat_zone" in info
        assert "vol_target" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = ZMomConfig(macd_fast=8, macd_slow=21, flat_zone=1.0)
        strategy = ZMomStrategy(config=config)
        assert strategy._config.macd_fast == 8
        assert strategy._config.flat_zone == 1.0

    def test_params_property(self) -> None:
        strategy = ZMomStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "macd_fast" in params
        assert "macd_slow" in params
        assert "flat_zone" in params

    def test_repr(self) -> None:
        strategy = ZMomStrategy()
        assert "z-mom" in strategy.name
        assert repr(strategy)  # truthy (not empty)

    def test_warmup_in_config(self) -> None:
        strategy = ZMomStrategy()
        warmup = strategy._config.warmup_periods()
        assert warmup > 0
