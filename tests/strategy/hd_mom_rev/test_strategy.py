"""Tests for hd-mom-rev strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.hd_mom_rev.config import HdMomRevConfig
from src.strategy.hd_mom_rev.strategy import HdMomRevStrategy


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
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


class TestRegistry:
    def test_registered(self) -> None:
        assert "hd-mom-rev" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("hd-mom-rev")
        assert cls is HdMomRevStrategy


class TestHdMomRevStrategy:
    def test_name(self) -> None:
        strategy = HdMomRevStrategy()
        assert strategy.name == "hd-mom-rev"

    def test_required_columns(self) -> None:
        strategy = HdMomRevStrategy()
        assert "close" in strategy.required_columns
        assert "open" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = HdMomRevStrategy()
        assert isinstance(strategy.config, HdMomRevConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = HdMomRevStrategy()
        result = strategy.preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = HdMomRevStrategy()
        df = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        strategy = HdMomRevStrategy()
        processed, signals = strategy.run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = HdMomRevStrategy.from_params(jump_threshold=1.5)
        assert isinstance(strategy, HdMomRevStrategy)

    def test_recommended_config(self) -> None:
        config = HdMomRevStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        strategy = HdMomRevStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0
        assert "jump_threshold" in info
        assert "confidence_cap" in info

    def test_custom_config(self) -> None:
        config = HdMomRevConfig(jump_threshold=1.5)
        strategy = HdMomRevStrategy(config=config)
        assert strategy._config.jump_threshold == 1.5

    def test_params_property(self) -> None:
        strategy = HdMomRevStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "jump_threshold" in params

    def test_repr(self) -> None:
        strategy = HdMomRevStrategy()
        assert "hd-mom-rev" in strategy.name
        assert repr(strategy)
