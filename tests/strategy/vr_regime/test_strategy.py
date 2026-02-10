"""Tests for VRRegimeStrategy."""

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.types import Direction
from src.strategy.vr_regime import VRRegimeConfig, VRRegimeStrategy


class TestRegistry:
    def test_registered(self):
        assert "vr-regime" in list_strategies()

    def test_get_strategy(self):
        assert get_strategy("vr-regime") == VRRegimeStrategy


class TestVRRegimeStrategy:
    def _make_sample_df(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 300
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
        strategy = VRRegimeStrategy()
        assert strategy.name == "VR-Regime"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert isinstance(strategy.config, VRRegimeConfig)

    def test_preprocess(self):
        df = self._make_sample_df()
        strategy = VRRegimeStrategy()
        processed = strategy.preprocess(df)

        expected_cols = [
            "returns",
            "vr",
            "vr_z_stat",
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
        strategy = VRRegimeStrategy()
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
        strategy = VRRegimeStrategy()
        processed_df, signals = strategy.run(df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self):
        df = self._make_sample_df()
        strategy = VRRegimeStrategy.from_params(
            vr_window=200,
            vr_k=10,
            mom_lookback=30,
        )
        assert strategy.config.vr_window == 200
        assert strategy.config.vr_k == 10

        _processed_df, signals = strategy.run(df)
        assert len(signals.entries) == len(df)

    def test_recommended_config(self):
        config = VRRegimeStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["use_trailing_stop"] is True

    def test_get_startup_info(self):
        strategy = VRRegimeStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "vr_window" in info
        assert "vr_k" in info
        assert "heteroscedastic" in info

    def test_warmup_periods(self):
        strategy = VRRegimeStrategy()
        # max(120 + 5, 20, 14) + 1 = 126
        assert strategy.warmup_periods() == 126

    def test_custom_config(self):
        config = VRRegimeConfig(vr_window=200, vr_k=8, mom_lookback=30)
        strategy = VRRegimeStrategy(config)
        assert strategy.config.vr_window == 200

    def test_params_property(self):
        strategy = VRRegimeStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "vr_window" in params
        assert "vr_k" in params

    def test_repr(self):
        strategy = VRRegimeStrategy()
        assert "VRRegimeStrategy" in repr(strategy)
