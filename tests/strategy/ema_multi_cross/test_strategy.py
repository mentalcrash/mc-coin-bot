"""Tests for EMA Multi-Cross strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.ema_multi_cross.config import EmaMultiCrossConfig
from src.strategy.ema_multi_cross.strategy import EmaMultiCrossStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "ema-multi-cross" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("ema-multi-cross")
        assert cls is EmaMultiCrossStrategy


class TestEmaMultiCrossStrategy:
    def test_name(self) -> None:
        assert EmaMultiCrossStrategy().name == "ema-multi-cross"

    def test_required_columns(self) -> None:
        strategy = EmaMultiCrossStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        assert isinstance(EmaMultiCrossStrategy().config, EmaMultiCrossConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = EmaMultiCrossStrategy().preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        s = EmaMultiCrossStrategy()
        df = s.preprocess(sample_ohlcv_df)
        signals = s.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        processed, signals = EmaMultiCrossStrategy().run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = EmaMultiCrossStrategy.from_params(pair1_fast=5, pair1_slow=15)
        assert isinstance(strategy, EmaMultiCrossStrategy)

    def test_recommended_config(self) -> None:
        config = EmaMultiCrossStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = EmaMultiCrossStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "pair1" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = EmaMultiCrossConfig(pair1_fast=5, pair1_slow=15)
        strategy = EmaMultiCrossStrategy(config=config)
        assert strategy._config.pair1_fast == 5

    def test_params_property(self) -> None:
        params = EmaMultiCrossStrategy().params
        assert isinstance(params, dict)
        assert "pair1_fast" in params

    def test_repr(self) -> None:
        strategy = EmaMultiCrossStrategy()
        assert "ema-multi-cross" in strategy.name
        assert repr(strategy)
