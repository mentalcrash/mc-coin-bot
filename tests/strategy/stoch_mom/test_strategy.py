"""Tests for StochMomStrategy (Integration)."""

import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.stoch_mom import StochMomConfig, StochMomStrategy


class TestRegistry:
    """Strategy Registry integration tests."""

    def test_strategy_registered(self) -> None:
        """'stoch-mom' is registered in the Registry."""
        available = list_strategies()
        assert "stoch-mom" in available

    def test_get_strategy(self) -> None:
        """get_strategy returns the correct class."""
        strategy_class = get_strategy("stoch-mom")
        assert strategy_class == StochMomStrategy


class TestStochMomStrategy:
    """StochMomStrategy class tests."""

    def test_strategy_properties(self) -> None:
        """Basic property verification."""
        strategy = StochMomStrategy()

        assert strategy.name == "Stoch-Mom"
        assert set(strategy.required_columns) == {"open", "high", "low", "close"}
        assert isinstance(strategy.config, StochMomConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """run() end-to-end pipeline."""
        strategy = StochMomStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # Preprocessed columns
        assert "pct_k" in processed_df.columns
        assert "pct_d" in processed_df.columns
        assert "sma" in processed_df.columns
        assert "atr" in processed_df.columns
        assert "vol_scalar" in processed_df.columns
        assert "vol_ratio" in processed_df.columns

        # Signal structure
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

        # Direction value range
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_from_params(self, sample_ohlcv: pd.DataFrame) -> None:
        """from_params() creates strategy (parameter sweep compatible)."""
        strategy = StochMomStrategy.from_params(
            k_period=10,
            sma_period=20,
            vol_target=0.30,
        )

        assert strategy.config.k_period == 10
        assert strategy.config.sma_period == 20
        assert strategy.config.vol_target == 0.30

        # Verify it runs
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self) -> None:
        """recommended_config values."""
        config = StochMomStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05

    def test_warmup_periods(self) -> None:
        """warmup_periods matches config."""
        strategy = StochMomStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()
        # Default: max(14, 30, 14) + 3 + 1 = 34
        assert warmup == 34

    def test_for_timeframe(self) -> None:
        """for_timeframe() factory."""
        strategy = StochMomStrategy.for_timeframe("1d")
        assert strategy.config.annualization_factor == 365.0

        strategy_4h = StochMomStrategy.for_timeframe("4h")
        assert strategy_4h.config.annualization_factor == 2190.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe() with additional parameters."""
        strategy = StochMomStrategy.for_timeframe("4h", vol_target=0.35)
        assert strategy.config.annualization_factor == 2190.0
        assert strategy.config.vol_target == 0.35

    def test_conservative_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """conservative() factory."""
        strategy = StochMomStrategy.conservative()
        assert strategy.config.k_period == 21
        assert strategy.config.sma_period == 50
        assert strategy.config.vol_target == 0.30
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_aggressive_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """aggressive() factory."""
        strategy = StochMomStrategy.aggressive()
        assert strategy.config.k_period == 9
        assert strategy.config.sma_period == 20
        assert strategy.config.vol_target == 0.50
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_get_startup_info(self) -> None:
        """get_startup_info keys."""
        strategy = StochMomStrategy()
        info = strategy.get_startup_info()

        assert "k_period" in info
        assert "d_period" in info
        assert "sma_period" in info
        assert "vol_target" in info
        assert "vol_ratio" in info
        assert "mode" in info

    def test_validate_input_missing_columns(self) -> None:
        """Missing required columns raise error."""
        strategy = StochMomStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)

    def test_validate_input_empty_df(self) -> None:
        """Empty DataFrame raises error."""
        strategy = StochMomStrategy()
        df = pd.DataFrame(
            columns=["open", "high", "low", "close"],
        )
        df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            strategy.run(df)

    def test_default_config(self) -> None:
        """Default config verification."""
        strategy = StochMomStrategy()
        assert strategy.config.k_period == 14
        assert strategy.config.d_period == 3
        assert strategy.config.sma_period == 30

    def test_custom_config(self) -> None:
        """Custom config creation."""
        config = StochMomConfig(k_period=10, vol_target=0.30)
        strategy = StochMomStrategy(config)
        assert strategy.config.k_period == 10
        assert strategy.config.vol_target == 0.30

    def test_params_dict(self) -> None:
        """params property matches config's model_dump."""
        strategy = StochMomStrategy()
        params = strategy.params
        assert params["k_period"] == 14
        assert params["d_period"] == 3
        assert params["sma_period"] == 30

    def test_repr(self) -> None:
        """__repr__ string verification."""
        strategy = StochMomStrategy()
        repr_str = repr(strategy)
        assert "StochMomStrategy" in repr_str
