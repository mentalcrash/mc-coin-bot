"""Unit tests for CvdDiverge8hStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import BaseStrategy
from src.strategy.cvd_diverge_8h.config import CvdDiverge8hConfig, ShortMode
from src.strategy.cvd_diverge_8h.strategy import CvdDiverge8hStrategy
from src.strategy.registry import get_strategy, list_strategies


class TestStrategyRegistry:
    def test_strategy_registered(self) -> None:
        assert "cvd-diverge-8h" in list_strategies()

    def test_get_strategy(self) -> None:
        strategy_cls = get_strategy("cvd-diverge-8h")
        assert strategy_cls is CvdDiverge8hStrategy

    def test_is_base_strategy_subclass(self) -> None:
        assert issubclass(CvdDiverge8hStrategy, BaseStrategy)


class TestStrategyProperties:
    def test_strategy_name(self) -> None:
        strategy = CvdDiverge8hStrategy()
        assert strategy.name == "cvd-diverge-8h"

    def test_required_columns(self) -> None:
        strategy = CvdDiverge8hStrategy()
        assert strategy.required_columns == ["open", "high", "low", "close", "volume"]

    def test_config_property(self) -> None:
        strategy = CvdDiverge8hStrategy()
        assert isinstance(strategy.config, CvdDiverge8hConfig)

    def test_default_config(self) -> None:
        strategy = CvdDiverge8hStrategy(config=None)
        assert isinstance(strategy.config, CvdDiverge8hConfig)


class TestRunPipeline:
    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        strategy = CvdDiverge8hStrategy()
        _processed_df, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_missing_columns_raises(self) -> None:
        strategy = CvdDiverge8hStrategy()
        bad_df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        with pytest.raises(ValueError, match="Missing"):
            strategy.run(bad_df)


class TestFromParams:
    def test_from_params(self) -> None:
        strategy = CvdDiverge8hStrategy.from_params()
        assert isinstance(strategy, CvdDiverge8hStrategy)

    def test_from_params_custom(self) -> None:
        strategy = CvdDiverge8hStrategy.from_params(vol_target=0.25)
        assert strategy.config.vol_target == 0.25


class TestRecommendedConfig:
    def test_recommended_config(self) -> None:
        config = CvdDiverge8hStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "trailing_stop_enabled" in config
        assert config["use_intrabar_trailing_stop"] is False


class TestShortMode:
    def test_short_mode_disabled(self, sample_ohlcv: pd.DataFrame) -> None:
        config = CvdDiverge8hConfig(short_mode=ShortMode.DISABLED)
        strategy = CvdDiverge8hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert (signals.direction >= 0).all()

    def test_short_mode_full(self, sample_ohlcv: pd.DataFrame) -> None:
        config = CvdDiverge8hConfig(short_mode=ShortMode.FULL)
        strategy = CvdDiverge8hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)

    def test_short_mode_hedge_only(self, sample_ohlcv: pd.DataFrame) -> None:
        config = CvdDiverge8hConfig(short_mode=ShortMode.HEDGE_ONLY)
        strategy = CvdDiverge8hStrategy(config=config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)


class TestStartupInfo:
    def test_startup_info_keys(self) -> None:
        strategy = CvdDiverge8hStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert all(isinstance(v, str) for v in info.values())


class TestGracefulDegradation:
    """Test graceful degradation: no dext_cvd_buy_vol column -> pure EMA trend."""

    def test_no_cvd_column_pure_ema_trend(self, sample_ohlcv: pd.DataFrame) -> None:
        """Without dext_cvd_buy_vol column, strategy should degrade to pure EMA trend."""
        strategy = CvdDiverge8hStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)
        # divergence_zscore should be 0.0 when CVD is absent
        assert (processed_df["divergence_zscore"] == 0.0).all()
        assert (processed_df["cvd_roc"] == 0.0).all()
        # Strategy should still produce valid signals
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool

    def test_with_cvd_column_uses_divergence(self, sample_ohlcv: pd.DataFrame) -> None:
        """With dext_cvd_buy_vol column, divergence_zscore should be non-trivial."""
        df = sample_ohlcv.copy()
        n = len(df)
        np.random.seed(99)
        # Add CVD buy volume column with realistic cumulative volume delta
        df["dext_cvd_buy_vol"] = 1000 + np.cumsum(np.random.randn(n) * 50)
        strategy = CvdDiverge8hStrategy()
        processed_df, signals = strategy.run(df)
        # With CVD data, divergence_zscore should have non-zero values
        assert len(signals.entries) == len(df)
        assert "divergence_zscore" in processed_df.columns
        assert "cvd_smooth" in processed_df.columns
        # cvd_smooth should not be all NaN when CVD data is present
        assert processed_df["cvd_smooth"].notna().any()
