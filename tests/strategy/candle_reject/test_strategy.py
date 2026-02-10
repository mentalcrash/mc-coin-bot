"""Tests for CandleRejectStrategy."""

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.candle_reject import CandleRejectConfig, CandleRejectStrategy
from src.strategy.types import Direction


class TestRegistry:
    def test_registered(self):
        assert "candle-reject" in list_strategies()

    def test_get_strategy(self):
        assert get_strategy("candle-reject") == CandleRejectStrategy


class TestCandleRejectStrategy:
    def _make_sample_df(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        open_ = close + np.random.randn(n) * 0.5
        bar_max = np.maximum(open_, close)
        bar_min = np.minimum(open_, close)
        high = bar_max + np.abs(np.random.randn(n) * 3.0)
        low = bar_min - np.abs(np.random.randn(n) * 3.0)
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
        strategy = CandleRejectStrategy()
        assert strategy.name == "Candle-Reject"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert isinstance(strategy.config, CandleRejectConfig)

    def test_preprocess(self):
        df = self._make_sample_df()
        strategy = CandleRejectStrategy()
        processed = strategy.preprocess(df)

        expected_cols = [
            "upper_wick",
            "lower_wick",
            "body",
            "range_",
            "bull_reject",
            "bear_reject",
            "body_position",
            "volume_zscore",
            "returns",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self):
        df = self._make_sample_df()
        strategy = CandleRejectStrategy()
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
        strategy = CandleRejectStrategy()
        processed_df, signals = strategy.run(df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self):
        df = self._make_sample_df()
        strategy = CandleRejectStrategy.from_params(
            rejection_threshold=0.7,
            volume_zscore_threshold=1.5,
            consecutive_boost=1.8,
        )
        assert strategy.config.rejection_threshold == 0.7
        assert strategy.config.volume_zscore_threshold == 1.5

        _processed_df, signals = strategy.run(df)
        assert len(signals.entries) == len(df)

    def test_recommended_config(self):
        config = CandleRejectStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["use_trailing_stop"] is True

    def test_get_startup_info(self):
        strategy = CandleRejectStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "rejection_threshold" in info
        assert "exit_timeout" in info
        assert "mode" in info

    def test_warmup_periods(self):
        strategy = CandleRejectStrategy()
        # max(30, 14) + 1 = 31
        assert strategy.warmup_periods() == 31

    def test_custom_config(self):
        config = CandleRejectConfig(rejection_threshold=0.7, volume_zscore_window=50)
        strategy = CandleRejectStrategy(config)
        assert strategy.config.rejection_threshold == 0.7

    def test_params_property(self):
        strategy = CandleRejectStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "rejection_threshold" in params
        assert "volume_zscore_threshold" in params

    def test_repr(self):
        strategy = CandleRejectStrategy()
        assert "CandleRejectStrategy" in repr(strategy)
