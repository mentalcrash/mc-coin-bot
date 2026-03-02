"""Tests for LR-Channel Multi-Scale Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.lr_channel_trend.config import LrChannelTrendConfig
from src.strategy.lr_channel_trend.strategy import LrChannelTrendStrategy


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
        assert "lr-channel-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("lr-channel-trend")
        assert cls is LrChannelTrendStrategy


class TestLrChannelTrendStrategy:
    def test_name(self) -> None:
        assert LrChannelTrendStrategy().name == "lr-channel-trend"

    def test_required_columns(self) -> None:
        cols = LrChannelTrendStrategy().required_columns
        for c in ["open", "high", "low", "close", "volume"]:
            assert c in cols

    def test_config(self) -> None:
        assert isinstance(LrChannelTrendStrategy().config, LrChannelTrendConfig)

    def test_custom_config(self) -> None:
        config = LrChannelTrendConfig(scale_short=10, scale_mid=40, scale_long=100)
        strategy = LrChannelTrendStrategy(config=config)
        assert strategy._config.scale_short == 10

    def test_default_config(self) -> None:
        strategy = LrChannelTrendStrategy(config=None)
        assert strategy._config.scale_short == 20
        assert strategy._config.scale_mid == 60
        assert strategy._config.scale_long == 150

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        result = LrChannelTrendStrategy().preprocess(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        s = LrChannelTrendStrategy()
        df = s.preprocess(sample_ohlcv_df)
        signals = s.generate_signals(df)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        processed, signals = LrChannelTrendStrategy().run(sample_ohlcv_df)
        assert len(processed) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self) -> None:
        strategy = LrChannelTrendStrategy.from_params(
            scale_short=10, scale_mid=40, scale_long=100
        )
        assert isinstance(strategy, LrChannelTrendStrategy)

    def test_recommended_config(self) -> None:
        config = LrChannelTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config

    def test_get_startup_info(self) -> None:
        info = LrChannelTrendStrategy().get_startup_info()
        assert isinstance(info, dict)
        assert "channel_multiplier" in info
        assert "scales" in info
        assert "vol_target" in info
        assert "short_mode" in info

    def test_get_startup_info_channel_multiplier_value(self) -> None:
        info = LrChannelTrendStrategy().get_startup_info()
        assert info["channel_multiplier"] == "2.0"

    def test_warmup_periods(self) -> None:
        strategy = LrChannelTrendStrategy()
        assert strategy._config.warmup_periods() >= strategy._config.scale_long
