"""Tests for ML Derivatives Regime strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.ml_deriv_regime.config import MlDerivRegimeConfig
from src.strategy.ml_deriv_regime.strategy import MlDerivRegimeStrategy


@pytest.fixture
def sample_ohlcv_with_funding_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.uniform(-0.001, 0.001, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "ml-deriv-regime" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("ml-deriv-regime")
        assert cls is MlDerivRegimeStrategy


class TestMlDerivRegimeStrategy:
    def test_name(self) -> None:
        strategy = MlDerivRegimeStrategy()
        assert strategy.name == "ml-deriv-regime"

    def test_required_columns(self) -> None:
        strategy = MlDerivRegimeStrategy()
        assert "close" in strategy.required_columns
        assert "funding_rate" in strategy.required_columns

    def test_config(self) -> None:
        strategy = MlDerivRegimeStrategy()
        assert isinstance(strategy.config, MlDerivRegimeConfig)

    def test_preprocess(self, sample_ohlcv_with_funding_df: pd.DataFrame) -> None:
        strategy = MlDerivRegimeStrategy()
        result = strategy.preprocess(sample_ohlcv_with_funding_df)
        assert len(result) == len(sample_ohlcv_with_funding_df)

    def test_generate_signals(self, sample_ohlcv_with_funding_df: pd.DataFrame) -> None:
        config = MlDerivRegimeConfig(training_window=60)
        strategy = MlDerivRegimeStrategy(config)
        df = strategy.preprocess(sample_ohlcv_with_funding_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_with_funding_df: pd.DataFrame) -> None:
        config = MlDerivRegimeConfig(training_window=60)
        strategy = MlDerivRegimeStrategy(config)
        processed, signals = strategy.run(sample_ohlcv_with_funding_df)
        assert len(processed) == len(sample_ohlcv_with_funding_df)
        assert len(signals.entries) == len(sample_ohlcv_with_funding_df)

    def test_from_params(self) -> None:
        strategy = MlDerivRegimeStrategy.from_params(training_window=126)
        assert isinstance(strategy, MlDerivRegimeStrategy)
        assert strategy._config.training_window == 126

    def test_recommended_config(self) -> None:
        config = MlDerivRegimeStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = MlDerivRegimeStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "training_window" in info

    def test_custom_config(self) -> None:
        config = MlDerivRegimeConfig(training_window=126)
        strategy = MlDerivRegimeStrategy(config=config)
        assert strategy._config.training_window == 126

    def test_params_property(self) -> None:
        strategy = MlDerivRegimeStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "training_window" in params

    def test_repr(self) -> None:
        strategy = MlDerivRegimeStrategy()
        assert "ml-deriv-regime" in strategy.name
        assert repr(strategy)

    def test_warmup_periods(self) -> None:
        strategy = MlDerivRegimeStrategy()
        assert strategy.warmup_periods() > 0

    def test_run_incremental(self, sample_ohlcv_with_funding_df: pd.DataFrame) -> None:
        config = MlDerivRegimeConfig(training_window=60)
        strategy = MlDerivRegimeStrategy(config)
        processed, _signals = strategy.run_incremental(sample_ohlcv_with_funding_df)
        assert len(processed) == len(sample_ohlcv_with_funding_df)
