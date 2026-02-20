"""Tests for EMA Cross Base strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.ema_cross_base.config import EmaCrossBaseConfig
from src.strategy.ema_cross_base.strategy import EmaCrossBaseStrategy


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
        assert "ema-cross-base" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("ema-cross-base")
        assert cls is EmaCrossBaseStrategy


class TestEmaCrossBaseStrategy:
    def test_name(self) -> None:
        assert EmaCrossBaseStrategy().name == "ema-cross-base"

    def test_required_columns(self) -> None:
        assert "close" in EmaCrossBaseStrategy().required_columns

    def test_config(self) -> None:
        assert isinstance(EmaCrossBaseStrategy().config, EmaCrossBaseConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = EmaCrossBaseStrategy().preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        s = EmaCrossBaseStrategy()
        df = s.preprocess(sample_ohlcv_df)
        signals = s.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        processed, signals = EmaCrossBaseStrategy().run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = EmaCrossBaseStrategy.from_params(fast_period=10, slow_period=50)
        assert isinstance(strategy, EmaCrossBaseStrategy)

    def test_recommended_config(self) -> None:
        config = EmaCrossBaseStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = EmaCrossBaseStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "fast_period" in info

    def test_params_property(self) -> None:
        params = EmaCrossBaseStrategy().params
        assert isinstance(params, dict)
        assert "fast_period" in params
