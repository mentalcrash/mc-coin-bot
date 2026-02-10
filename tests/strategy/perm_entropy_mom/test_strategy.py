"""Tests for PermEntropyMomStrategy."""

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.perm_entropy_mom import PermEntropyMomConfig, PermEntropyMomStrategy
from src.strategy.types import Direction


class TestRegistry:
    def test_registered(self):
        assert "perm-entropy-mom" in list_strategies()

    def test_get_strategy(self):
        assert get_strategy("perm-entropy-mom") == PermEntropyMomStrategy


class TestPermEntropyMomStrategy:
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
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

    def test_properties(self):
        strategy = PermEntropyMomStrategy()
        assert strategy.name == "Perm-Entropy-Mom"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert isinstance(strategy.config, PermEntropyMomConfig)

    def test_preprocess(self):
        df = self._make_sample_df()
        strategy = PermEntropyMomStrategy()
        processed = strategy.preprocess(df)

        expected_cols = [
            "returns",
            "pe_short",
            "pe_long",
            "mom_direction",
            "realized_vol",
            "vol_scalar",
            "conviction",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self):
        df = self._make_sample_df()
        strategy = PermEntropyMomStrategy()
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
        strategy = PermEntropyMomStrategy()
        processed_df, signals = strategy.run(df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self):
        df = self._make_sample_df()
        strategy = PermEntropyMomStrategy.from_params(
            pe_order=4,
            pe_short_window=20,
            pe_long_window=50,
            mom_lookback=25,
        )
        assert strategy.config.pe_order == 4
        assert strategy.config.pe_short_window == 20

        _processed_df, signals = strategy.run(df)
        assert len(signals.entries) == len(df)

    def test_recommended_config(self):
        config = PermEntropyMomStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["use_trailing_stop"] is True

    def test_get_startup_info(self):
        strategy = PermEntropyMomStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "pe_order" in info
        assert "pe_short_window" in info
        assert "pe_long_window" in info
        assert "noise_threshold" in info
        assert "mode" in info

    def test_warmup_periods(self):
        strategy = PermEntropyMomStrategy()
        # max(60 + 3, 30, 14) + 1 = 64
        assert strategy.warmup_periods() == 64

    def test_custom_config(self):
        config = PermEntropyMomConfig(
            pe_order=4,
            pe_short_window=40,
            pe_long_window=80,
        )
        strategy = PermEntropyMomStrategy(config)
        assert strategy.config.pe_order == 4

    def test_params_property(self):
        strategy = PermEntropyMomStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "pe_order" in params
        assert "pe_short_window" in params
        assert "noise_threshold" in params

    def test_repr(self):
        strategy = PermEntropyMomStrategy()
        assert "PermEntropyMomStrategy" in repr(strategy)

    def test_startup_info_hedge_mode(self):
        strategy = PermEntropyMomStrategy()
        info = strategy.get_startup_info()
        assert "Hedge-Short" in info["mode"]

    def test_startup_info_full_mode(self):
        from src.strategy.perm_entropy_mom.config import ShortMode

        config = PermEntropyMomConfig(short_mode=ShortMode.FULL)
        strategy = PermEntropyMomStrategy(config)
        info = strategy.get_startup_info()
        assert info["mode"] == "Long/Short"
