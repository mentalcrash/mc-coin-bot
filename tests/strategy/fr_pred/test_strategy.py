"""Tests for FR-Pred strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.fr_pred.config import FRPredConfig
from src.strategy.fr_pred.strategy import FRPredStrategy


@pytest.fixture
def sample_deriv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.randn(n) * 0.0005
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "fr-pred" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("fr-pred")
        assert cls is FRPredStrategy


class TestFRPredStrategy:
    def test_name(self) -> None:
        strategy = FRPredStrategy()
        assert strategy.name == "fr-pred"

    def test_required_columns(self) -> None:
        strategy = FRPredStrategy()
        assert "close" in strategy.required_columns
        assert "funding_rate" in strategy.required_columns

    def test_config(self) -> None:
        strategy = FRPredStrategy()
        assert isinstance(strategy.config, FRPredConfig)

    def test_preprocess(self, sample_deriv_df: pd.DataFrame) -> None:
        strategy = FRPredStrategy()
        result = strategy.preprocess(sample_deriv_df)
        assert len(result) == len(sample_deriv_df)

    def test_generate_signals(self, sample_deriv_df: pd.DataFrame) -> None:
        strategy = FRPredStrategy()
        df = strategy.preprocess(sample_deriv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_deriv_df: pd.DataFrame) -> None:
        strategy = FRPredStrategy()
        processed, signals = strategy.run(sample_deriv_df)
        assert len(processed) == len(sample_deriv_df)
        assert len(signals.entries) == len(sample_deriv_df)

    def test_from_params(self) -> None:
        strategy = FRPredStrategy.from_params(fr_mr_threshold=1.5)
        assert isinstance(strategy, FRPredStrategy)

    def test_recommended_config(self) -> None:
        config = FRPredStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = FRPredStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "fr_mr_threshold" in info

    def test_custom_config(self) -> None:
        config = FRPredConfig(fr_mr_threshold=1.5)
        strategy = FRPredStrategy(config=config)
        assert strategy._config.fr_mr_threshold == 1.5  # noqa: SLF001

    def test_params_property(self) -> None:
        strategy = FRPredStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "fr_ma_window" in params

    def test_repr(self) -> None:
        strategy = FRPredStrategy()
        assert "fr-pred" in strategy.name
        assert repr(strategy)
