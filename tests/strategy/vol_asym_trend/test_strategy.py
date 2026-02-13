"""Tests for vol-asym-trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.vol_asym_trend.config import VolAsymTrendConfig
from src.strategy.vol_asym_trend.strategy import VolAsymTrendStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="6h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "vol-asym-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("vol-asym-trend")
        assert cls is VolAsymTrendStrategy


class TestVolAsymTrendStrategy:
    def test_name(self) -> None:
        strategy = VolAsymTrendStrategy()
        assert strategy.name == "vol-asym-trend"

    def test_required_columns(self) -> None:
        strategy = VolAsymTrendStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = VolAsymTrendStrategy()
        assert isinstance(strategy.config, VolAsymTrendConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = VolAsymTrendStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = VolAsymTrendStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = VolAsymTrendStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = VolAsymTrendStrategy.from_params(asym_window=20)
        assert isinstance(strategy, VolAsymTrendStrategy)

    def test_recommended_config(self) -> None:
        config = VolAsymTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = VolAsymTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_custom_config(self) -> None:
        config = VolAsymTrendConfig(asym_window=20)
        strategy = VolAsymTrendStrategy(config=config)
        assert strategy._config.asym_window == 20

    def test_params_property(self) -> None:
        strategy = VolAsymTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "asym_window" in params

    def test_repr(self) -> None:
        strategy = VolAsymTrendStrategy()
        assert "vol-asym-trend" in strategy.name
        assert repr(strategy)
