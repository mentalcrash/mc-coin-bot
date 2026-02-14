"""Tests for Regime-Gated Multi-Factor MR strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.regime_mf_mr.config import RegimeMfMrConfig
from src.strategy.regime_mf_mr.strategy import RegimeMfMrStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "regime-mf-mr" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("regime-mf-mr")
        assert cls is RegimeMfMrStrategy


class TestRegimeMfMrStrategy:
    def test_name(self) -> None:
        strategy = RegimeMfMrStrategy()
        assert strategy.name == "regime-mf-mr"

    def test_required_columns(self) -> None:
        strategy = RegimeMfMrStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = RegimeMfMrStrategy()
        assert isinstance(strategy.config, RegimeMfMrConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = RegimeMfMrStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = RegimeMfMrStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = RegimeMfMrStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = RegimeMfMrStrategy.from_params(bb_period=30)
        assert isinstance(strategy, RegimeMfMrStrategy)
        assert strategy._config.bb_period == 30

    def test_recommended_config(self) -> None:
        config = RegimeMfMrStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = RegimeMfMrStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "bb_period" in info

    def test_custom_config(self) -> None:
        config = RegimeMfMrConfig(bb_period=30)
        strategy = RegimeMfMrStrategy(config=config)
        assert strategy._config.bb_period == 30

    def test_params_property(self) -> None:
        strategy = RegimeMfMrStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "bb_period" in params

    def test_repr(self) -> None:
        strategy = RegimeMfMrStrategy()
        assert "regime-mf-mr" in strategy.name
        assert repr(strategy)

    def test_warmup_periods(self) -> None:
        strategy = RegimeMfMrStrategy()
        assert strategy.warmup_periods() > 0
