"""Tests for Carry-Momentum Convergence strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.carry_mom_convergence.config import CarryMomConvergenceConfig
from src.strategy.carry_mom_convergence.strategy import CarryMomConvergenceStrategy


@pytest.fixture
def sample_ohlcv_with_fr_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.uniform(-0.001, 0.001, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "carry-mom-convergence" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("carry-mom-convergence")
        assert cls is CarryMomConvergenceStrategy


class TestCarryMomConvergenceStrategy:
    def test_name(self) -> None:
        strategy = CarryMomConvergenceStrategy()
        assert strategy.name == "carry-mom-convergence"

    def test_required_columns(self) -> None:
        strategy = CarryMomConvergenceStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "funding_rate" in strategy.required_columns

    def test_config(self) -> None:
        strategy = CarryMomConvergenceStrategy()
        assert isinstance(strategy.config, CarryMomConvergenceConfig)

    def test_preprocess(self, sample_ohlcv_with_fr_df: pd.DataFrame) -> None:
        strategy = CarryMomConvergenceStrategy()
        result = strategy.preprocess(sample_ohlcv_with_fr_df)
        assert len(result) == len(sample_ohlcv_with_fr_df)

    def test_generate_signals(self, sample_ohlcv_with_fr_df: pd.DataFrame) -> None:
        strategy = CarryMomConvergenceStrategy()
        df = strategy.preprocess(sample_ohlcv_with_fr_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_with_fr_df: pd.DataFrame) -> None:
        strategy = CarryMomConvergenceStrategy()
        processed, signals = strategy.run(sample_ohlcv_with_fr_df)
        assert len(processed) == len(sample_ohlcv_with_fr_df)
        assert len(signals.entries) == len(sample_ohlcv_with_fr_df)

    def test_from_params(self) -> None:
        strategy = CarryMomConvergenceStrategy.from_params(mom_lookback=30)
        assert isinstance(strategy, CarryMomConvergenceStrategy)

    def test_recommended_config(self) -> None:
        config = CarryMomConvergenceStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = CarryMomConvergenceStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "mom_lookback" in info

    def test_custom_config(self) -> None:
        config = CarryMomConvergenceConfig(mom_lookback=30, fr_lookback=5)
        strategy = CarryMomConvergenceStrategy(config=config)
        assert strategy._config.mom_lookback == 30

    def test_params_property(self) -> None:
        strategy = CarryMomConvergenceStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "mom_lookback" in params

    def test_repr(self) -> None:
        strategy = CarryMomConvergenceStrategy()
        assert "carry-mom-convergence" in strategy.name
        assert repr(strategy)
