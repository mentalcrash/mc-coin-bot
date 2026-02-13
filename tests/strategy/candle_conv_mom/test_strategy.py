"""Tests for candle-conv-mom strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.candle_conv_mom.config import CandleConvMomConfig
from src.strategy.candle_conv_mom.strategy import CandleConvMomStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "candle-conv-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("candle-conv-mom")
        assert cls is CandleConvMomStrategy


class TestCandleConvMomStrategy:
    def test_name(self) -> None:
        strategy = CandleConvMomStrategy()
        assert strategy.name == "candle-conv-mom"

    def test_required_columns(self) -> None:
        strategy = CandleConvMomStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = CandleConvMomStrategy()
        assert isinstance(strategy.config, CandleConvMomConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CandleConvMomStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CandleConvMomStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CandleConvMomStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = CandleConvMomStrategy.from_params(conv_window=10)
        assert isinstance(strategy, CandleConvMomStrategy)

    def test_recommended_config(self) -> None:
        config = CandleConvMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = CandleConvMomStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_custom_config(self) -> None:
        config = CandleConvMomConfig(conv_window=10)
        strategy = CandleConvMomStrategy(config=config)
        assert strategy._config.conv_window == 10

    def test_params_property(self) -> None:
        strategy = CandleConvMomStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "conv_window" in params

    def test_repr(self) -> None:
        strategy = CandleConvMomStrategy()
        assert "candle-conv-mom" in strategy.name
        assert repr(strategy)
