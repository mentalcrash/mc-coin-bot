"""Tests for Vol Squeeze Breakout strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.vol_squeeze_brk.config import VolSqueezeBrkConfig
from src.strategy.vol_squeeze_brk.strategy import VolSqueezeBrkStrategy


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
        assert "vol-squeeze-brk" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("vol-squeeze-brk")
        assert cls is VolSqueezeBrkStrategy


class TestVolSqueezeBrkStrategy:
    def test_name(self) -> None:
        assert VolSqueezeBrkStrategy().name == "vol-squeeze-brk"

    def test_required_columns(self) -> None:
        assert "close" in VolSqueezeBrkStrategy().required_columns

    def test_config(self) -> None:
        assert isinstance(VolSqueezeBrkStrategy().config, VolSqueezeBrkConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = VolSqueezeBrkStrategy().preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        s = VolSqueezeBrkStrategy()
        df = s.preprocess(sample_ohlcv_df)
        signals = s.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        processed, signals = VolSqueezeBrkStrategy().run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = VolSqueezeBrkStrategy.from_params(bb_period=30)
        assert isinstance(strategy, VolSqueezeBrkStrategy)

    def test_recommended_config(self) -> None:
        config = VolSqueezeBrkStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = VolSqueezeBrkStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_custom_config(self) -> None:
        config = VolSqueezeBrkConfig(bb_period=30)
        strategy = VolSqueezeBrkStrategy(config=config)
        assert strategy._config.bb_period == 30

    def test_params_property(self) -> None:
        params = VolSqueezeBrkStrategy().params
        assert isinstance(params, dict)
        assert "bb_period" in params

    def test_repr(self) -> None:
        assert repr(VolSqueezeBrkStrategy())
