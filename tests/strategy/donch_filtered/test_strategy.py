"""Tests for Donchian Filtered strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.donch_filtered.config import DonchFilteredConfig
from src.strategy.donch_filtered.strategy import DonchFilteredStrategy


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
        assert "donch-filtered" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("donch-filtered")
        assert cls is DonchFilteredStrategy


class TestDonchFilteredStrategy:
    def test_name(self) -> None:
        strategy = DonchFilteredStrategy()
        assert strategy.name == "donch-filtered"

    def test_required_columns(self) -> None:
        strategy = DonchFilteredStrategy()
        assert "close" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = DonchFilteredStrategy()
        assert isinstance(strategy.config, DonchFilteredConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = DonchFilteredStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = DonchFilteredStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = DonchFilteredStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = DonchFilteredStrategy.from_params(
            lookback_short=10, lookback_mid=30, lookback_long=60, fr_suppress_threshold=2.0
        )
        assert isinstance(strategy, DonchFilteredStrategy)

    def test_recommended_config(self) -> None:
        config = DonchFilteredStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config
        assert config["use_intrabar_trailing_stop"] is False

    def test_get_startup_info(self) -> None:
        strategy = DonchFilteredStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "lookbacks" in info
        assert "fr_suppress_threshold" in info

    def test_custom_config(self) -> None:
        config = DonchFilteredConfig(
            lookback_short=10, lookback_mid=30, lookback_long=60, fr_suppress_threshold=2.0
        )
        strategy = DonchFilteredStrategy(config=config)
        assert strategy._config.lookback_short == 10
        assert strategy._config.fr_suppress_threshold == 2.0

    def test_params_property(self) -> None:
        strategy = DonchFilteredStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "lookback_short" in params
        assert "fr_suppress_threshold" in params

    def test_repr(self) -> None:
        strategy = DonchFilteredStrategy()
        assert "donch-filtered" in strategy.name
        assert repr(strategy)
