"""Tests for HMMRegimeStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.hmm_regime import HMMRegimeConfig, HMMRegimeStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Sample OHLCV DataFrame (300 days) with regime structure."""
    np.random.seed(42)
    n = 300

    returns = np.zeros(n)
    returns[:100] = np.random.randn(100) * 0.02 + 0.003
    returns[100:200] = np.random.randn(100) * 0.025 - 0.004
    returns[200:] = np.random.randn(100) * 0.02 + 0.003

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    return df


@pytest.fixture
def small_config() -> HMMRegimeConfig:
    """Small config for fast tests."""
    return HMMRegimeConfig(
        n_states=3,
        n_iter=50,
        min_train_window=100,
        retrain_interval=50,
        vol_window=20,
    )


class TestRegistry:
    """Strategy Registry tests."""

    def test_registered(self) -> None:
        """'hmm-regime' is registered in the strategy registry."""
        available = list_strategies()
        assert "hmm-regime" in available

    def test_get_strategy(self) -> None:
        """get_strategy() returns HMMRegimeStrategy class."""
        strategy_class = get_strategy("hmm-regime")
        assert strategy_class == HMMRegimeStrategy


class TestHMMRegimeStrategy:
    """HMMRegimeStrategy tests."""

    def test_properties(self) -> None:
        """Strategy property tests."""
        strategy = HMMRegimeStrategy()

        assert strategy.name == "HMM-Regime"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, HMMRegimeConfig)

    def test_preprocess(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HMMRegimeConfig,
    ) -> None:
        """preprocess() test with small config."""
        strategy = HMMRegimeStrategy(small_config)
        processed = strategy.preprocess(sample_ohlcv_df)

        expected_cols = [
            "returns",
            "rolling_vol",
            "regime",
            "regime_prob",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns, f"Missing column: {col}"

    def test_generate_signals(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HMMRegimeConfig,
    ) -> None:
        """generate_signals() test."""
        strategy = HMMRegimeStrategy(small_config)
        processed = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(processed)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_run_pipeline(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HMMRegimeConfig,
    ) -> None:
        """Full pipeline (run) test."""
        strategy = HMMRegimeStrategy(small_config)
        processed_df, signals = strategy.run(sample_ohlcv_df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """from_params() strategy creation."""
        strategy = HMMRegimeStrategy.from_params(
            n_states=2,
            n_iter=50,
            min_train_window=100,
            retrain_interval=50,
        )

        assert strategy.config.n_states == 2
        assert strategy.config.n_iter == 50
        assert strategy.config.min_train_window == 100

        # Full pipeline should work
        _processed_df, signals = strategy.run(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_recommended_config(self) -> None:
        """recommended_config() test."""
        config = HMMRegimeStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10

    def test_get_startup_info(self) -> None:
        """get_startup_info() test."""
        strategy = HMMRegimeStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "n_states" in info
        assert "min_train_window" in info
        assert "retrain_interval" in info
        assert "n_iter" in info
        assert "vol_target" in info
        assert "mode" in info

        # Default mode should be Long-Only
        assert info["mode"] == "Long-Only"

    def test_warmup_periods(self) -> None:
        """warmup_periods() test."""
        strategy = HMMRegimeStrategy()
        warmup = strategy.warmup_periods()

        # Default: min_train_window=252 + 1 = 253
        assert warmup == 253

    def test_params_property(self) -> None:
        """params property returns config dict."""
        strategy = HMMRegimeStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "n_states" in params
        assert "min_train_window" in params
        assert "retrain_interval" in params
        assert "n_iter" in params
        assert "vol_target" in params

    def test_repr(self) -> None:
        """String representation test."""
        strategy = HMMRegimeStrategy()
        repr_str = repr(strategy)

        assert "HMMRegimeStrategy" in repr_str
