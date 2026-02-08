"""Tests for OvernightStrategy (full pipeline)."""

import pandas as pd

from src.strategy.overnight.config import OvernightConfig
from src.strategy.overnight.strategy import OvernightStrategy
from src.strategy.registry import get_strategy, list_strategies
from src.strategy.types import StrategySignals


class TestStrategyRegistration:
    """Registry integration tests."""

    def test_strategy_registered(self) -> None:
        """'overnight' is in list_strategies()."""
        names = list_strategies()
        assert "overnight" in names

    def test_get_strategy(self) -> None:
        """get_strategy('overnight') returns OvernightStrategy class."""
        cls = get_strategy("overnight")
        assert cls is OvernightStrategy


class TestStrategyProperties:
    """Property tests."""

    def test_strategy_properties(self) -> None:
        """name='Overnight', required_columns includes OHLCV."""
        strategy = OvernightStrategy()

        assert strategy.name == "Overnight"
        assert "open" in strategy.required_columns
        assert "high" in strategy.required_columns
        assert "low" in strategy.required_columns
        assert "close" in strategy.required_columns
        assert "volume" in strategy.required_columns

    def test_config_property(self) -> None:
        """config returns OvernightConfig instance."""
        strategy = OvernightStrategy()
        assert isinstance(strategy.config, OvernightConfig)

    def test_custom_config(self) -> None:
        """Custom config is stored correctly."""
        config = OvernightConfig(entry_hour=20, exit_hour=6, vol_target=0.25)
        strategy = OvernightStrategy(config=config)

        assert strategy.config.entry_hour == 20
        assert strategy.config.exit_hour == 6
        assert strategy.config.vol_target == 0.25


class TestStrategyPipeline:
    """Full pipeline tests."""

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """Full run() pipeline returns processed DataFrame and signals."""
        strategy = OvernightStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # Processed DataFrame has indicator columns
        assert "hour" in processed_df.columns
        assert "returns" in processed_df.columns
        assert "realized_vol" in processed_df.columns
        assert "vol_scalar" in processed_df.columns
        assert "atr" in processed_df.columns

        # Signals are valid
        assert isinstance(signals, StrategySignals)
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_pipeline_with_custom_config(self, sample_ohlcv: pd.DataFrame) -> None:
        """Pipeline works with custom config."""
        config = OvernightConfig(entry_hour=18, exit_hour=6, vol_target=0.20)
        strategy = OvernightStrategy(config=config)
        _processed_df, signals = strategy.run(sample_ohlcv)

        assert isinstance(signals, StrategySignals)
        assert (signals.direction == 1).any()


class TestStrategyFactoryMethods:
    """Factory method tests."""

    def test_from_params(self) -> None:
        """from_params creates strategy with correct config."""
        strategy = OvernightStrategy.from_params(
            entry_hour=20,
            exit_hour=4,
            vol_target=0.25,
        )

        assert isinstance(strategy, OvernightStrategy)
        assert strategy.config.entry_hour == 20
        assert strategy.config.exit_hour == 4
        assert strategy.config.vol_target == 0.25

    def test_from_params_default(self) -> None:
        """from_params with no args uses defaults."""
        strategy = OvernightStrategy.from_params()
        assert strategy.config.entry_hour == 22
        assert strategy.config.exit_hour == 0

    def test_recommended_config(self) -> None:
        """recommended_config returns expected PM settings."""
        rec = OvernightStrategy.recommended_config()

        assert isinstance(rec, dict)
        assert rec["max_leverage_cap"] == 1.5
        assert rec["system_stop_loss"] == 0.10
        assert rec["rebalance_threshold"] == 0.05
        assert rec["use_trailing_stop"] is False


class TestStrategyWarmup:
    """Warmup period tests."""

    def test_warmup_periods(self) -> None:
        """warmup_periods delegates to config.warmup_periods()."""
        strategy = OvernightStrategy()
        assert strategy.warmup_periods() == strategy.config.warmup_periods()
        # Default: vol_window=30, so warmup = 31
        assert strategy.warmup_periods() == 31

    def test_warmup_periods_custom(self) -> None:
        """Custom vol_window changes warmup."""
        config = OvernightConfig(vol_window=48)
        strategy = OvernightStrategy(config=config)
        assert strategy.warmup_periods() == 49


class TestStrategyStartupInfo:
    """get_startup_info tests."""

    def test_startup_info_keys(self) -> None:
        """get_startup_info returns expected keys."""
        strategy = OvernightStrategy()
        info = strategy.get_startup_info()

        assert "session" in info
        assert "vol_target" in info
        assert "vol_window" in info
        assert "mode" in info

    def test_startup_info_values(self) -> None:
        """get_startup_info returns formatted values."""
        strategy = OvernightStrategy()
        info = strategy.get_startup_info()

        assert info["session"] == "22:00-00:00 UTC"
        assert info["mode"] == "Long-Only"

    def test_startup_info_with_vol_filter(self) -> None:
        """Vol filter adds vol_filter key to startup info."""
        config = OvernightConfig(use_vol_filter=True)
        strategy = OvernightStrategy(config=config)
        info = strategy.get_startup_info()

        assert "vol_filter" in info
