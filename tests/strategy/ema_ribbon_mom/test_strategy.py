"""Tests for EMA Ribbon Momentum strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.ema_ribbon_mom.config import EmaRibbonMomConfig
from src.strategy.ema_ribbon_mom.strategy import EmaRibbonMomStrategy


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
        assert "ema-ribbon-momentum" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("ema-ribbon-momentum")
        assert cls is EmaRibbonMomStrategy


class TestEmaRibbonMomStrategy:
    def test_name(self) -> None:
        assert EmaRibbonMomStrategy().name == "ema-ribbon-momentum"

    def test_required_columns(self) -> None:
        strategy = EmaRibbonMomStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        assert isinstance(EmaRibbonMomStrategy().config, EmaRibbonMomConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = EmaRibbonMomStrategy().preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        s = EmaRibbonMomStrategy()
        df = s.preprocess(sample_ohlcv_df)
        signals = s.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        processed, signals = EmaRibbonMomStrategy().run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = EmaRibbonMomStrategy.from_params(roc_period=14)
        assert isinstance(strategy, EmaRibbonMomStrategy)

    def test_recommended_config(self) -> None:
        config = EmaRibbonMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = EmaRibbonMomStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "ema_periods" in info
        assert "short_mode" in info

    def test_custom_config(self) -> None:
        config = EmaRibbonMomConfig(roc_period=14)
        strategy = EmaRibbonMomStrategy(config=config)
        assert strategy._config.roc_period == 14

    def test_params_property(self) -> None:
        params = EmaRibbonMomStrategy().params
        assert isinstance(params, dict)
        assert "ema_periods" in params

    def test_repr(self) -> None:
        strategy = EmaRibbonMomStrategy()
        assert "ema-ribbon-momentum" in strategy.name
        assert repr(strategy)
