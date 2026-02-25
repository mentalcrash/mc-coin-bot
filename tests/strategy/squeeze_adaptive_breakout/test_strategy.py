"""Tests for Squeeze-Adaptive Breakout strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.squeeze_adaptive_breakout.config import SqueezeAdaptiveBreakoutConfig
from src.strategy.squeeze_adaptive_breakout.strategy import SqueezeAdaptiveBreakoutStrategy


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
        assert "squeeze-adaptive-breakout" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("squeeze-adaptive-breakout")
        assert cls is SqueezeAdaptiveBreakoutStrategy


class TestSqueezeAdaptiveBreakoutStrategy:
    def test_name(self) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy()
        assert strategy.name == "squeeze-adaptive-breakout"

    def test_required_columns(self) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns

    def test_config(self) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy()
        assert isinstance(strategy.config, SqueezeAdaptiveBreakoutConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy.from_params(bb_period=15, kc_mult=2.0)
        assert isinstance(strategy, SqueezeAdaptiveBreakoutStrategy)
        assert strategy._config.bb_period == 15
        assert strategy._config.kc_mult == 2.0

    def test_recommended_config(self) -> None:
        config = SqueezeAdaptiveBreakoutStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "bb" in info
        assert "kama" in info
        assert "vol_target" in info

    def test_custom_config(self) -> None:
        config = SqueezeAdaptiveBreakoutConfig(bb_period=15, squeeze_lookback=5)
        strategy = SqueezeAdaptiveBreakoutStrategy(config=config)
        assert strategy._config.bb_period == 15
        assert strategy._config.squeeze_lookback == 5

    def test_params_property(self) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "bb_period" in params
        assert "kama_er_lookback" in params
        assert "squeeze_lookback" in params

    def test_repr(self) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy()
        assert "squeeze-adaptive-breakout" in strategy.name
        assert repr(strategy)  # truthy (not empty)

    def test_default_config_none(self) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy(config=None)
        assert strategy.config.bb_period == 20

    def test_run_with_validate_input(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """validate_input -> preprocess -> generate_signals pipeline."""
        strategy = SqueezeAdaptiveBreakoutStrategy()
        strategy.validate_input(sample_ohlcv_df)
        processed = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(processed)
        assert len(signals.direction) == len(sample_ohlcv_df)

    @pytest.mark.parametrize("bb_period", [5, 10, 20, 30, 50])
    def test_various_bb_periods(self, sample_ohlcv_df: pd.DataFrame, bb_period: int) -> None:
        strategy = SqueezeAdaptiveBreakoutStrategy.from_params(bb_period=bb_period)
        _processed, signals = strategy.run(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)
