"""Tests for MAD-Channel Multi-Scale Trend strategy integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.mad_channel_trend.config import MadChannelTrendConfig
from src.strategy.mad_channel_trend.strategy import MadChannelTrendStrategy


class _OhlcvMixin:
    @staticmethod
    def _make_ohlcv(n: int = 300) -> pd.DataFrame:
        np.random.seed(42)
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
        assert "mad-channel-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        cls = get_strategy("mad-channel-trend")
        assert cls is MadChannelTrendStrategy


class TestMadChannelTrendStrategy(_OhlcvMixin):
    def test_name(self) -> None:
        strategy = MadChannelTrendStrategy()
        assert strategy.name == "mad-channel-trend"

    def test_required_columns(self) -> None:
        strategy = MadChannelTrendStrategy()
        assert "close" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config(self) -> None:
        strategy = MadChannelTrendStrategy()
        assert isinstance(strategy.config, MadChannelTrendConfig)

    def test_preprocess(self) -> None:
        strategy = MadChannelTrendStrategy()
        df = self._make_ohlcv()
        result = strategy.preprocess(df)
        assert len(result) == len(df)

    def test_generate_signals(self) -> None:
        strategy = MadChannelTrendStrategy()
        df = self._make_ohlcv()
        processed = strategy.preprocess(df)
        signals = strategy.generate_signals(processed)
        assert len(signals.entries) == len(df)

    def test_run_pipeline(self) -> None:
        strategy = MadChannelTrendStrategy()
        df = self._make_ohlcv()
        processed, signals = strategy.run(df)
        assert len(processed) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self) -> None:
        strategy = MadChannelTrendStrategy.from_params(scale_short=10, scale_mid=40, scale_long=100)
        assert isinstance(strategy, MadChannelTrendStrategy)

    def test_recommended_config(self) -> None:
        config = MadChannelTrendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "stop_loss_pct" in config
        assert "trailing_stop_enabled" in config

    def test_get_startup_info(self) -> None:
        strategy = MadChannelTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "scales" in info
        assert "entry_threshold" in info
        assert "short_mode" in info
        assert "mad_multiplier" in info
        assert "keltner_multiplier" in info

    def test_custom_config(self) -> None:
        config = MadChannelTrendConfig(scale_short=10, scale_mid=40, scale_long=100)
        strategy = MadChannelTrendStrategy(config=config)
        assert strategy._config.scale_short == 10

    def test_params_property(self) -> None:
        strategy = MadChannelTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "scale_short" in params
        assert "scale_mid" in params
        assert "scale_long" in params

    def test_repr(self) -> None:
        strategy = MadChannelTrendStrategy()
        assert "mad-channel-trend" in strategy.name
        assert repr(strategy)
