"""Tests for OUMeanRevStrategy."""

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.ou_meanrev import OUMeanRevConfig, OUMeanRevStrategy
from src.strategy.types import Direction


class TestRegistry:
    def test_registered(self):
        assert "ou-meanrev" in list_strategies()

    def test_get_strategy(self):
        assert get_strategy("ou-meanrev") == OUMeanRevStrategy


class TestOUMeanRevStrategy:
    def _make_sample_df(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 300
        prices = np.zeros(n)
        prices[0] = 100.0
        mu = 100.0
        theta = 0.05
        sigma = 2.0
        for i in range(1, n):
            prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + sigma * np.random.randn()

        high = prices + np.abs(np.random.randn(n) * 1.5)
        low = prices - np.abs(np.random.randn(n) * 1.5)
        open_ = prices + np.random.randn(n) * 0.5
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": prices,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

    def test_properties(self):
        strategy = OUMeanRevStrategy()
        assert strategy.name == "OU-MeanRev"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert isinstance(strategy.config, OUMeanRevConfig)

    def test_preprocess(self):
        df = self._make_sample_df()
        strategy = OUMeanRevStrategy()
        processed = strategy.preprocess(df)

        expected_cols = [
            "returns",
            "theta",
            "half_life",
            "ou_mu",
            "ou_zscore",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self):
        df = self._make_sample_df()
        strategy = OUMeanRevStrategy()
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
        strategy = OUMeanRevStrategy()
        processed_df, signals = strategy.run(df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self):
        df = self._make_sample_df()
        strategy = OUMeanRevStrategy.from_params(
            ou_window=80,
            entry_zscore=1.5,
            max_half_life=20,
        )
        assert strategy.config.ou_window == 80
        assert strategy.config.entry_zscore == 1.5

        _processed_df, signals = strategy.run(df)
        assert len(signals.entries) == len(df)

    def test_recommended_config(self):
        config = OUMeanRevStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["use_trailing_stop"] is True

    def test_get_startup_info(self):
        strategy = OUMeanRevStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "ou_window" in info
        assert "entry_zscore" in info
        assert "max_half_life" in info
        assert "timeout" in info
        assert "mode" in info

    def test_warmup_periods(self):
        strategy = OUMeanRevStrategy()
        # max(120, 20, 14) + 1 = 121
        assert strategy.warmup_periods() == 121

    def test_custom_config(self):
        config = OUMeanRevConfig(ou_window=200, entry_zscore=3.0, mom_lookback=30)
        strategy = OUMeanRevStrategy(config)
        assert strategy.config.ou_window == 200

    def test_params_property(self):
        strategy = OUMeanRevStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "ou_window" in params
        assert "entry_zscore" in params
        assert "max_half_life" in params

    def test_repr(self):
        strategy = OUMeanRevStrategy()
        assert "OUMeanRevStrategy" in repr(strategy)
