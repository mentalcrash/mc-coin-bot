"""Tests for Regime-Adaptive Multi-Lookback Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.regime_adaptive_mom.config import RegimeAdaptiveMomConfig
from src.strategy.regime_adaptive_mom.strategy import RegimeAdaptiveMomStrategy


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
        assert "regime-adaptive-momentum" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("regime-adaptive-momentum")
        assert cls is RegimeAdaptiveMomStrategy


class TestRegimeAdaptiveMomStrategy:
    def test_name(self) -> None:
        strategy = RegimeAdaptiveMomStrategy()
        assert strategy.name == "regime-adaptive-momentum"

    def test_required_columns(self) -> None:
        strategy = RegimeAdaptiveMomStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = RegimeAdaptiveMomStrategy()
        assert isinstance(strategy.config, RegimeAdaptiveMomConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = RegimeAdaptiveMomStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = RegimeAdaptiveMomStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = RegimeAdaptiveMomStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = RegimeAdaptiveMomStrategy.from_params(
            fast_lookback=10, mid_lookback=40, slow_lookback=80
        )
        assert isinstance(strategy, RegimeAdaptiveMomStrategy)

    def test_recommended_config(self) -> None:
        config = RegimeAdaptiveMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = RegimeAdaptiveMomStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "lookbacks" in info

    def test_custom_config(self) -> None:
        config = RegimeAdaptiveMomConfig(fast_lookback=10, mid_lookback=40, slow_lookback=80)
        strategy = RegimeAdaptiveMomStrategy(config=config)
        assert strategy._config.fast_lookback == 10

    def test_params_property(self) -> None:
        strategy = RegimeAdaptiveMomStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "fast_lookback" in params

    def test_repr(self) -> None:
        strategy = RegimeAdaptiveMomStrategy()
        assert "regime-adaptive-momentum" in strategy.name
        assert repr(strategy)
