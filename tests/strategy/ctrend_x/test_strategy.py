"""Tests for CTREND-X strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.ctrend_x.config import CTRENDXConfig
from src.strategy.ctrend_x.strategy import CTRENDXStrategy


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
        assert "ctrend-x" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("ctrend-x")
        assert cls is CTRENDXStrategy


class TestCTRENDXStrategy:
    def test_name(self) -> None:
        strategy = CTRENDXStrategy()
        assert strategy.name == "ctrend-x"

    def test_required_columns(self) -> None:
        strategy = CTRENDXStrategy()
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = CTRENDXStrategy()
        assert isinstance(strategy.config, CTRENDXConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CTRENDXStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CTRENDXStrategy(
            config=CTRENDXConfig(training_window=60, n_estimators=10, max_depth=2)
        )
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = CTRENDXStrategy(
            config=CTRENDXConfig(training_window=60, n_estimators=10, max_depth=2)
        )
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = CTRENDXStrategy.from_params(training_window=120)
        assert isinstance(strategy, CTRENDXStrategy)

    def test_recommended_config(self) -> None:
        config = CTRENDXStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = CTRENDXStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "n_estimators" in info

    def test_custom_config(self) -> None:
        config = CTRENDXConfig(n_estimators=50)
        strategy = CTRENDXStrategy(config=config)
        assert strategy._config.n_estimators == 50  # noqa: SLF001

    def test_params_property(self) -> None:
        strategy = CTRENDXStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "training_window" in params

    def test_repr(self) -> None:
        strategy = CTRENDXStrategy()
        assert "ctrend-x" in strategy.name
        assert repr(strategy)
