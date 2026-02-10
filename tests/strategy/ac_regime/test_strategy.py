"""Tests for ACRegimeStrategy."""

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.ac_regime import ACRegimeConfig, ACRegimeStrategy
from src.strategy.types import Direction


class TestRegistry:
    def test_registered(self):
        assert "ac-regime" in list_strategies()

    def test_get_strategy(self):
        assert get_strategy("ac-regime") == ACRegimeStrategy


class TestACRegimeStrategy:
    def _make_sample_df(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)
        open_ = close + np.random.randn(n) * 0.5
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

    def test_properties(self):
        strategy = ACRegimeStrategy()
        assert strategy.name == "AC-Regime"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert isinstance(strategy.config, ACRegimeConfig)

    def test_preprocess(self):
        df = self._make_sample_df()
        strategy = ACRegimeStrategy()
        processed = strategy.preprocess(df)

        expected_cols = [
            "returns",
            "ac_rho",
            "sig_bound",
            "mom_direction",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self):
        df = self._make_sample_df()
        strategy = ACRegimeStrategy()
        processed = strategy.preprocess(df)
        signals = strategy.generate_signals(processed)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_run_pipeline(self):
        df = self._make_sample_df()
        strategy = ACRegimeStrategy()
        processed_df, signals = strategy.run(df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self):
        df = self._make_sample_df()
        strategy = ACRegimeStrategy.from_params(
            ac_window=40,
            ac_lag=2,
            mom_lookback=30,
        )
        assert strategy.config.ac_window == 40
        assert strategy.config.ac_lag == 2
        assert strategy.config.mom_lookback == 30

        _processed_df, signals = strategy.run(df)
        assert len(signals.entries) == len(df)

    def test_recommended_config(self):
        config = ACRegimeStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self):
        strategy = ACRegimeStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "ac_window" in info
        assert "ac_lag" in info
        assert "significance_z" in info
        assert "mode" in info

    def test_warmup_periods(self):
        strategy = ACRegimeStrategy()
        # max(60 + 1, 20, 14) + 1 = 62
        assert strategy.warmup_periods() == 62

    def test_custom_config(self):
        config = ACRegimeConfig(ac_window=100, mom_lookback=40)
        strategy = ACRegimeStrategy(config)
        assert strategy.config.ac_window == 100

    def test_params_property(self):
        strategy = ACRegimeStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "ac_window" in params
        assert "ac_lag" in params

    def test_repr(self):
        strategy = ACRegimeStrategy()
        assert "ACRegimeStrategy" in repr(strategy)
