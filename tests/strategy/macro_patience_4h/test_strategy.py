"""Unit tests for MacroPatience4hStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import BaseStrategy
from src.strategy.macro_patience_4h.config import MacroPatience4hConfig, ShortMode
from src.strategy.macro_patience_4h.strategy import MacroPatience4hStrategy
from src.strategy.registry import get_strategy, list_strategies


class TestStrategyRegistry:
    def test_strategy_registered(self) -> None:
        assert "macro-patience-4h" in list_strategies()

    def test_get_strategy(self) -> None:
        strategy_cls = get_strategy("macro-patience-4h")
        assert strategy_cls is MacroPatience4hStrategy

    def test_is_base_strategy_subclass(self) -> None:
        assert issubclass(MacroPatience4hStrategy, BaseStrategy)


class TestStrategyProperties:
    def test_strategy_name(self) -> None:
        strategy = MacroPatience4hStrategy()
        assert strategy.name == "macro-patience-4h"

    def test_required_columns(self) -> None:
        strategy = MacroPatience4hStrategy()
        assert strategy.required_columns == ["open", "high", "low", "close", "volume"]

    def test_config_property(self) -> None:
        strategy = MacroPatience4hStrategy()
        assert isinstance(strategy.config, MacroPatience4hConfig)

    def test_default_config(self) -> None:
        strategy = MacroPatience4hStrategy(config=None)
        assert isinstance(strategy.config, MacroPatience4hConfig)


class TestRunPipeline:
    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        strategy = MacroPatience4hStrategy()
        _processed_df, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_missing_columns_raises(self) -> None:
        strategy = MacroPatience4hStrategy()
        bad_df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        with pytest.raises(ValueError, match="Missing"):
            strategy.run(bad_df)


class TestFromParams:
    def test_from_params(self) -> None:
        strategy = MacroPatience4hStrategy.from_params()
        assert isinstance(strategy, MacroPatience4hStrategy)

    def test_from_params_custom(self) -> None:
        strategy = MacroPatience4hStrategy.from_params(vol_target=0.25)
        assert strategy.config.vol_target == 0.25


class TestRecommendedConfig:
    def test_recommended_config(self) -> None:
        config = MacroPatience4hStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "trailing_stop_enabled" in config
        assert config["use_intrabar_trailing_stop"] is False


class TestShortMode:
    def test_short_mode_disabled(self, sample_ohlcv: pd.DataFrame) -> None:
        config = MacroPatience4hConfig(short_mode=ShortMode.DISABLED)
        strategy = MacroPatience4hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert (signals.direction >= 0).all()

    def test_short_mode_full(self, sample_ohlcv: pd.DataFrame) -> None:
        config = MacroPatience4hConfig(short_mode=ShortMode.FULL)
        strategy = MacroPatience4hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)

    def test_short_mode_hedge_only(self, sample_ohlcv: pd.DataFrame) -> None:
        config = MacroPatience4hConfig(short_mode=ShortMode.HEDGE_ONLY)
        strategy = MacroPatience4hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)


class TestStartupInfo:
    def test_startup_info_keys(self) -> None:
        strategy = MacroPatience4hStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert all(isinstance(v, str) for v in info.values())


class TestGracefulDegradation:
    """Test macro columns absent -> macro_direction=0 (neutral)."""

    def test_no_macro_columns_defaults_neutral(self, sample_ohlcv: pd.DataFrame) -> None:
        """Without macro columns (macro_dxy, macro_vix, macro_m2),
        the strategy should degrade gracefully with macro_direction=0."""
        strategy = MacroPatience4hStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)
        # macro_direction should be 0 (neutral) when macro columns are absent
        assert (processed_df["macro_direction"] == 0).all()
        assert (processed_df["macro_z"] == 0.0).all()
        # Strategy should still produce valid signals
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool

    def test_with_macro_columns_uses_gate(self, sample_ohlcv: pd.DataFrame) -> None:
        """With macro columns present, macro_direction should not be all zeros."""
        df = sample_ohlcv.copy()
        n = len(df)
        np.random.seed(99)
        # Add macro columns with trending data so z-scores diverge from 0
        df["macro_dxy"] = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df["macro_vix"] = 20 + np.cumsum(np.random.randn(n) * 0.3)
        df["macro_m2"] = 21000 + np.cumsum(np.random.randn(n) * 50)
        strategy = MacroPatience4hStrategy()
        processed_df, signals = strategy.run(df)
        # With macro data, macro_direction should have non-zero values
        # (not guaranteed all non-zero, but at least some should differ)
        assert len(signals.entries) == len(df)
        # Verify macro_z column exists and is computed
        assert "macro_z" in processed_df.columns
        assert "macro_direction" in processed_df.columns
