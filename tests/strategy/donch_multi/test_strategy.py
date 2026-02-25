"""Tests for Donchian Multi-Scale strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.donch_multi.config import DonchMultiConfig
from src.strategy.donch_multi.strategy import DonchMultiStrategy


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
        assert "donch-multi" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("donch-multi")
        assert cls is DonchMultiStrategy


class TestDonchMultiStrategy:
    def test_name(self) -> None:
        strategy = DonchMultiStrategy()
        assert strategy.name == "donch-multi"

    def test_required_columns(self) -> None:
        strategy = DonchMultiStrategy()
        assert "close" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = DonchMultiStrategy()
        assert isinstance(strategy.config, DonchMultiConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = DonchMultiStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = DonchMultiStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = DonchMultiStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = DonchMultiStrategy.from_params(
            lookback_short=10, lookback_mid=30, lookback_long=60
        )
        assert isinstance(strategy, DonchMultiStrategy)

    def test_recommended_config(self) -> None:
        config = DonchMultiStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = DonchMultiStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "lookbacks" in info
        assert "entry_threshold" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = DonchMultiConfig(lookback_short=10, lookback_mid=30, lookback_long=60)
        strategy = DonchMultiStrategy(config=config)
        assert strategy._config.lookback_short == 10

    def test_params_property(self) -> None:
        strategy = DonchMultiStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "lookback_short" in params

    def test_repr(self) -> None:
        strategy = DonchMultiStrategy()
        assert "donch-multi" in strategy.name
        assert repr(strategy)
