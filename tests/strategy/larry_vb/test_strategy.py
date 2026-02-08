"""Tests for LarryVBStrategy (full pipeline)."""

import pandas as pd

from src.strategy.larry_vb.config import LarryVBConfig, ShortMode
from src.strategy.larry_vb.strategy import LarryVBStrategy
from src.strategy.registry import get_strategy, list_strategies
from src.strategy.types import StrategySignals


class TestStrategyRegistry:
    """Registry integration tests."""

    def test_strategy_registered(self) -> None:
        """'larry-vb' is in list_strategies()."""
        names = list_strategies()
        assert "larry-vb" in names

    def test_get_strategy(self) -> None:
        """get_strategy('larry-vb') returns LarryVBStrategy class."""
        cls = get_strategy("larry-vb")
        assert cls is LarryVBStrategy


class TestStrategyProperties:
    """Property tests."""

    def test_strategy_name(self) -> None:
        """name='Larry-VB'."""
        strategy = LarryVBStrategy()
        assert strategy.name == "Larry-VB"

    def test_required_columns(self) -> None:
        """required_columns includes OHLC."""
        strategy = LarryVBStrategy()
        assert "open" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns
        assert "close" in strategy.required_columns

    def test_config_property(self) -> None:
        """config returns LarryVBConfig instance."""
        strategy = LarryVBStrategy()
        assert isinstance(strategy.config, LarryVBConfig)

    def test_custom_config(self) -> None:
        """Custom config is stored correctly."""
        config = LarryVBConfig(k_factor=0.7, vol_target=0.30)
        strategy = LarryVBStrategy(config=config)

        assert strategy.config.k_factor == 0.7
        assert strategy.config.vol_target == 0.30

    def test_default_config_when_none(self) -> None:
        """config=None uses default LarryVBConfig."""
        strategy = LarryVBStrategy(config=None)
        assert strategy.config.k_factor == 0.5
        assert strategy.config.short_mode == ShortMode.FULL


class TestRunPipeline:
    """Full pipeline tests."""

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """Full run() pipeline returns processed DataFrame and signals."""
        strategy = LarryVBStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # Processed DataFrame has indicator columns
        assert "prev_range" in processed_df.columns
        assert "breakout_upper" in processed_df.columns
        assert "breakout_lower" in processed_df.columns
        assert "realized_vol" in processed_df.columns
        assert "vol_scalar" in processed_df.columns

        # Signals are valid
        assert isinstance(signals, StrategySignals)
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_pipeline_with_custom_config(self, sample_ohlcv: pd.DataFrame) -> None:
        """Pipeline works with custom config."""
        config = LarryVBConfig(k_factor=0.3, vol_target=0.50)
        strategy = LarryVBStrategy(config=config)
        _processed_df, signals = strategy.run(sample_ohlcv)

        assert isinstance(signals, StrategySignals)
        # Aggressive config (low k) should produce more breakout signals
        assert len(signals.entries) == len(sample_ohlcv)

    def test_run_pipeline_disabled_short(self, sample_ohlcv: pd.DataFrame) -> None:
        """Pipeline with DISABLED short mode produces no -1 direction."""
        config = LarryVBConfig(short_mode=ShortMode.DISABLED)
        strategy = LarryVBStrategy(config=config)
        _processed_df, signals = strategy.run(sample_ohlcv)

        assert -1 not in signals.direction.values


class TestFromParams:
    """from_params factory method tests."""

    def test_from_params(self) -> None:
        """from_params creates strategy with correct config."""
        strategy = LarryVBStrategy.from_params(
            k_factor=0.7,
            vol_target=0.30,
            vol_window=30,
        )

        assert isinstance(strategy, LarryVBStrategy)
        assert strategy.config.k_factor == 0.7
        assert strategy.config.vol_target == 0.30
        assert strategy.config.vol_window == 30

    def test_from_params_default(self) -> None:
        """from_params with no args uses defaults."""
        strategy = LarryVBStrategy.from_params()
        assert strategy.config.k_factor == 0.5
        assert strategy.config.vol_target == 0.40


class TestRecommendedConfig:
    """recommended_config tests."""

    def test_recommended_config(self) -> None:
        """recommended_config returns expected PM settings."""
        rec = LarryVBStrategy.recommended_config()

        assert isinstance(rec, dict)
        assert rec["max_leverage_cap"] == 2.0
        assert rec["system_stop_loss"] == 0.10
        assert rec["rebalance_threshold"] == 0.05


class TestWarmupPeriods:
    """Warmup period tests."""

    def test_warmup_periods_default(self) -> None:
        """warmup_periods delegates to config.warmup_periods()."""
        strategy = LarryVBStrategy()
        assert strategy.warmup_periods() == strategy.config.warmup_periods()
        # Default: vol_window=20, so warmup = 22
        assert strategy.warmup_periods() == 22

    def test_warmup_periods_custom(self) -> None:
        """Custom vol_window changes warmup."""
        config = LarryVBConfig(vol_window=50)
        strategy = LarryVBStrategy(config=config)
        assert strategy.warmup_periods() == 52


class TestTimeframeFactory:
    """for_timeframe + strategy tests."""

    def test_strategy_with_timeframe_config(self) -> None:
        """Strategy works with for_timeframe config."""
        config = LarryVBConfig.for_timeframe("1d")
        strategy = LarryVBStrategy(config=config)
        assert strategy.config.annualization_factor == 365.0

    def test_strategy_with_4h_timeframe(self) -> None:
        """Strategy with 4h timeframe has correct annualization."""
        config = LarryVBConfig.for_timeframe("4h")
        strategy = LarryVBStrategy(config=config)
        assert strategy.config.annualization_factor == 2190.0


class TestStartupInfo:
    """get_startup_info tests."""

    def test_startup_info_keys(self) -> None:
        """get_startup_info returns expected keys."""
        strategy = LarryVBStrategy()
        info = strategy.get_startup_info()

        assert "k_factor" in info
        assert "vol_window" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_startup_info_values(self) -> None:
        """get_startup_info returns formatted values."""
        strategy = LarryVBStrategy()
        info = strategy.get_startup_info()

        assert info["k_factor"] == "0.50"
        assert info["vol_window"] == "20d"
        assert info["vol_target"] == "40%"
        assert info["mode"] == "Long/Short"

    def test_startup_info_disabled_mode(self) -> None:
        """Disabled mode shows 'Long-Only'."""
        config = LarryVBConfig(short_mode=ShortMode.DISABLED)
        strategy = LarryVBStrategy(config=config)
        info = strategy.get_startup_info()

        assert info["mode"] == "Long-Only"
