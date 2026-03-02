"""Tests for Carry-Regime Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.carry_regime_12h.config import CarryRegimeConfig
from src.strategy.carry_regime_12h.strategy import CarryRegimeStrategy


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
        assert "carry-regime-12h" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("carry-regime-12h")
        assert cls is CarryRegimeStrategy


class TestCarryRegimeStrategy:
    def test_name(self) -> None:
        strategy = CarryRegimeStrategy()
        assert strategy.name == "carry-regime-12h"

    def test_required_columns(self) -> None:
        strategy = CarryRegimeStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = CarryRegimeStrategy()
        assert isinstance(strategy.config, CarryRegimeConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CarryRegimeStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CarryRegimeStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CarryRegimeStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = CarryRegimeStrategy.from_params(ema_fast=5, ema_mid=15, ema_slow=40)
        assert isinstance(strategy, CarryRegimeStrategy)

    def test_recommended_config(self) -> None:
        config = CarryRegimeStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = CarryRegimeStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "ema_fast" in info
        assert "carry_sensitivity" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = CarryRegimeConfig(ema_fast=5, ema_mid=15, ema_slow=40)
        strategy = CarryRegimeStrategy(config=config)
        assert strategy._config.ema_fast == 5

    def test_params_property(self) -> None:
        strategy = CarryRegimeStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "ema_fast" in params

    def test_repr(self) -> None:
        strategy = CarryRegimeStrategy()
        assert "carry-regime-12h" in strategy.name
        assert repr(strategy)

    def test_with_funding_rate_data(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """funding_rate 데이터 포함 시 정상 동작."""
        np.random.seed(123)
        df = sample_ohlcv_df.copy()
        df["funding_rate"] = np.random.uniform(-0.001, 0.001, len(df))
        strategy = CarryRegimeStrategy()
        _processed, signals = strategy.run(df)
        assert len(signals.entries) == len(df)
