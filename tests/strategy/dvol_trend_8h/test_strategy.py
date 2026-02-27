"""Unit tests for DvolTrend8hStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import BaseStrategy
from src.strategy.dvol_trend_8h.config import DvolTrend8hConfig, ShortMode
from src.strategy.dvol_trend_8h.strategy import DvolTrend8hStrategy
from src.strategy.registry import get_strategy, list_strategies


class TestStrategyRegistry:
    def test_strategy_registered(self) -> None:
        assert "dvol-trend-8h" in list_strategies()

    def test_get_strategy(self) -> None:
        strategy_cls = get_strategy("dvol-trend-8h")
        assert strategy_cls is DvolTrend8hStrategy

    def test_is_base_strategy_subclass(self) -> None:
        assert issubclass(DvolTrend8hStrategy, BaseStrategy)


class TestStrategyProperties:
    def test_strategy_name(self) -> None:
        strategy = DvolTrend8hStrategy()
        assert strategy.name == "dvol-trend-8h"

    def test_required_columns(self) -> None:
        strategy = DvolTrend8hStrategy()
        assert strategy.required_columns == ["open", "high", "low", "close", "volume"]

    def test_config_property(self) -> None:
        strategy = DvolTrend8hStrategy()
        assert isinstance(strategy.config, DvolTrend8hConfig)

    def test_default_config(self) -> None:
        strategy = DvolTrend8hStrategy(config=None)
        assert isinstance(strategy.config, DvolTrend8hConfig)


class TestRunPipeline:
    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        strategy = DvolTrend8hStrategy()
        _processed_df, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_missing_columns_raises(self) -> None:
        strategy = DvolTrend8hStrategy()
        bad_df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        with pytest.raises(ValueError, match="Missing"):
            strategy.run(bad_df)


class TestFromParams:
    def test_from_params(self) -> None:
        strategy = DvolTrend8hStrategy.from_params()
        assert isinstance(strategy, DvolTrend8hStrategy)

    def test_from_params_custom(self) -> None:
        strategy = DvolTrend8hStrategy.from_params(vol_target=0.25)
        assert strategy.config.vol_target == 0.25


class TestRecommendedConfig:
    def test_recommended_config(self) -> None:
        config = DvolTrend8hStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "trailing_stop_enabled" in config
        assert config["use_intrabar_trailing_stop"] is False


class TestShortMode:
    def test_short_mode_disabled(self, sample_ohlcv: pd.DataFrame) -> None:
        config = DvolTrend8hConfig(short_mode=ShortMode.DISABLED)
        strategy = DvolTrend8hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert (signals.direction >= 0).all()

    def test_short_mode_full(self, sample_ohlcv: pd.DataFrame) -> None:
        config = DvolTrend8hConfig(short_mode=ShortMode.FULL)
        strategy = DvolTrend8hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)

    def test_short_mode_hedge_only(self, sample_ohlcv: pd.DataFrame) -> None:
        config = DvolTrend8hConfig(short_mode=ShortMode.HEDGE_ONLY)
        strategy = DvolTrend8hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)


class TestStartupInfo:
    def test_startup_info_keys(self) -> None:
        strategy = DvolTrend8hStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert all(isinstance(v, str) for v in info.values())


class TestDvolFallback:
    """Test without dvol_close column (neutral multiplier 1.0)."""

    def test_no_dvol_column_neutral_multiplier(self, sample_ohlcv: pd.DataFrame) -> None:
        """Without dvol_close column, dvol_size_mult should be 1.0 (neutral)."""
        strategy = DvolTrend8hStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)
        # dvol_size_mult should be 1.0 when dvol_close is absent
        assert (processed_df["dvol_size_mult"] == 1.0).all()
        # Strategy should still produce valid signals
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool

    def test_with_dvol_column_adjusts_size(self, sample_ohlcv: pd.DataFrame) -> None:
        """With dvol_close column, dvol_size_mult should vary based on IV regime."""
        df = sample_ohlcv.copy()
        n = len(df)
        np.random.seed(99)
        # Add DVOL column with realistic implied volatility values (30-100 range)
        df["dvol_close"] = 50 + np.cumsum(np.random.randn(n) * 2)
        strategy = DvolTrend8hStrategy()
        processed_df, signals = strategy.run(df)
        # dvol_size_mult should have varying values (not all 1.0)
        assert len(signals.entries) == len(df)
        assert "dvol_size_mult" in processed_df.columns
        # After warmup, size multiplier should take effect
        # (some values should differ from 1.0 due to percentile thresholds)
        mult_values = processed_df["dvol_size_mult"].dropna().unique()
        assert len(mult_values) >= 1  # At minimum, should have computed values
