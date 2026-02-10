"""Tests for VPINFlowStrategy."""

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.types import Direction
from src.strategy.vpin_flow import VPINFlowConfig, VPINFlowStrategy


class TestRegistry:
    def test_registered(self):
        assert "vpin-flow" in list_strategies()

    def test_get_strategy(self):
        assert get_strategy("vpin-flow") == VPINFlowStrategy


class TestVPINFlowStrategy:
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
        strategy = VPINFlowStrategy()
        assert strategy.name == "VPIN-Flow"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert isinstance(strategy.config, VPINFlowConfig)

    def test_preprocess(self):
        df = self._make_sample_df()
        strategy = VPINFlowStrategy()
        processed = strategy.preprocess(df)

        expected_cols = [
            "returns",
            "v_buy",
            "v_sell",
            "vpin",
            "flow_direction",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self):
        df = self._make_sample_df()
        strategy = VPINFlowStrategy()
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
        strategy = VPINFlowStrategy()
        processed_df, signals = strategy.run(df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self):
        df = self._make_sample_df()
        strategy = VPINFlowStrategy.from_params(
            n_buckets=30,
            threshold_high=0.8,
            threshold_low=0.2,
        )
        assert strategy.config.n_buckets == 30
        assert strategy.config.threshold_high == 0.8

        _processed_df, signals = strategy.run(df)
        assert len(signals.entries) == len(df)

    def test_recommended_config(self):
        config = VPINFlowStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["use_trailing_stop"] is True

    def test_get_startup_info(self):
        strategy = VPINFlowStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "n_buckets" in info
        assert "threshold_high" in info
        assert "threshold_low" in info

    def test_warmup_periods(self):
        strategy = VPINFlowStrategy()
        # max(50, 20, 14) + 1 = 51
        assert strategy.warmup_periods() == 51

    def test_custom_config(self):
        config = VPINFlowConfig(n_buckets=80, threshold_high=0.8, threshold_low=0.2)
        strategy = VPINFlowStrategy(config)
        assert strategy.config.n_buckets == 80

    def test_params_property(self):
        strategy = VPINFlowStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "n_buckets" in params
        assert "threshold_high" in params

    def test_repr(self):
        strategy = VPINFlowStrategy()
        assert "VPINFlowStrategy" in repr(strategy)
