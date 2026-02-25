"""Tests for MVRV Cycle Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.mvrv_cycle_trend.config import MvrvCycleTrendConfig
from src.strategy.mvrv_cycle_trend.strategy import MvrvCycleTrendStrategy


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
        assert "mvrv-cycle-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("mvrv-cycle-trend")
        assert cls is MvrvCycleTrendStrategy


class TestMvrvCycleTrendStrategy:
    def test_name(self) -> None:
        strategy = MvrvCycleTrendStrategy()
        assert strategy.name == "mvrv-cycle-trend"

    def test_required_columns(self) -> None:
        strategy = MvrvCycleTrendStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns
        assert "open" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns

    def test_config(self) -> None:
        strategy = MvrvCycleTrendStrategy()
        assert isinstance(strategy.config, MvrvCycleTrendConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = MvrvCycleTrendStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = MvrvCycleTrendStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = MvrvCycleTrendStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = MvrvCycleTrendStrategy.from_params(mom_fast=10, mom_slow=40)
        assert isinstance(strategy, MvrvCycleTrendStrategy)
        assert strategy._config.mom_fast == 10
        assert strategy._config.mom_slow == 40

    def test_recommended_config(self) -> None:
        config = MvrvCycleTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = MvrvCycleTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "mvrv_bull" in info
        assert "mvrv_bear" in info
        assert "mom_fast" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = MvrvCycleTrendConfig(mom_fast=10, mom_slow=40)
        strategy = MvrvCycleTrendStrategy(config=config)
        assert strategy._config.mom_fast == 10

    def test_params_property(self) -> None:
        strategy = MvrvCycleTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "mom_fast" in params
        assert "mom_slow" in params
        assert "mvrv_bull_threshold" in params

    def test_repr(self) -> None:
        strategy = MvrvCycleTrendStrategy()
        assert "mvrv-cycle-trend" in strategy.name
        assert repr(strategy)  # truthy (not empty)

    def test_default_config_none(self) -> None:
        strategy = MvrvCycleTrendStrategy(config=None)
        assert isinstance(strategy.config, MvrvCycleTrendConfig)

    def test_with_onchain_data(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """On-chain 데이터 포함 시에도 정상 동작."""
        df = sample_ohlcv_df.copy()
        df["oc_mvrv"] = np.random.uniform(0.5, 4.0, len(df))
        strategy = MvrvCycleTrendStrategy(config=MvrvCycleTrendConfig(mvrv_zscore_window=90))
        _processed, signals = strategy.run(df)
        assert len(signals.entries) == len(df)
