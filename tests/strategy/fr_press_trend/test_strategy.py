"""Tests for Funding Pressure Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.fr_press_trend.config import FrPressTrendConfig
from src.strategy.fr_press_trend.strategy import FrPressTrendStrategy


@pytest.fixture
def sample_ohlcv_fr_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.randn(n) * 0.0003
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "fr-press-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("fr-press-trend")
        assert cls is FrPressTrendStrategy


class TestFrPressTrendStrategy:
    def test_name(self) -> None:
        assert FrPressTrendStrategy().name == "fr-press-trend"

    def test_required_columns(self) -> None:
        cols = FrPressTrendStrategy().required_columns
        assert "close" in cols
        assert "funding_rate" in cols

    def test_config(self) -> None:
        assert isinstance(FrPressTrendStrategy().config, FrPressTrendConfig)

    def test_preprocess(self, sample_ohlcv_fr_df: pd.DataFrame) -> None:
        result = FrPressTrendStrategy().preprocess(sample_ohlcv_fr_df)
        assert len(result) == len(sample_ohlcv_fr_df)

    def test_generate_signals(self, sample_ohlcv_fr_df: pd.DataFrame) -> None:
        s = FrPressTrendStrategy()
        df = s.preprocess(sample_ohlcv_fr_df)
        signals = s.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_fr_df: pd.DataFrame) -> None:
        processed, signals = FrPressTrendStrategy().run(sample_ohlcv_fr_df)
        assert len(processed) == len(sample_ohlcv_fr_df)
        assert len(signals.entries) == len(sample_ohlcv_fr_df)

    def test_from_params(self) -> None:
        strategy = FrPressTrendStrategy.from_params(sma_fast=15)
        assert isinstance(strategy, FrPressTrendStrategy)

    def test_recommended_config(self) -> None:
        config = FrPressTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = FrPressTrendStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_custom_config(self) -> None:
        config = FrPressTrendConfig(sma_fast=15)
        strategy = FrPressTrendStrategy(config=config)
        assert strategy._config.sma_fast == 15

    def test_params_property(self) -> None:
        params = FrPressTrendStrategy().params
        assert isinstance(params, dict)
        assert "sma_fast" in params

    def test_repr(self) -> None:
        assert repr(FrPressTrendStrategy())
