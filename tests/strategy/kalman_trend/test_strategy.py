"""Tests for KalmanTrendStrategy."""

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.kalman_trend import KalmanTrendConfig, KalmanTrendStrategy
from src.strategy.types import Direction


class TestRegistry:
    def test_registered(self) -> None:
        assert "kalman-trend" in list_strategies()

    def test_get_strategy(self) -> None:
        assert get_strategy("kalman-trend") == KalmanTrendStrategy


class TestKalmanTrendStrategy:
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

    def test_properties(self) -> None:
        strategy = KalmanTrendStrategy()
        assert strategy.name == "Kalman-Trend"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert isinstance(strategy.config, KalmanTrendConfig)

    def test_preprocess(self) -> None:
        df = self._make_sample_df()
        strategy = KalmanTrendStrategy()
        processed = strategy.preprocess(df)

        expected_cols = [
            "returns",
            "kalman_state",
            "kalman_velocity",
            "q_adaptive",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self) -> None:
        df = self._make_sample_df()
        strategy = KalmanTrendStrategy()
        processed = strategy.preprocess(df)
        signals = strategy.generate_signals(processed)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_run_pipeline(self) -> None:
        df = self._make_sample_df()
        strategy = KalmanTrendStrategy()
        processed_df, signals = strategy.run(df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self) -> None:
        df = self._make_sample_df()
        strategy = KalmanTrendStrategy.from_params(
            base_q=0.05,
            observation_noise=2.0,
            vol_lookback=10,
            long_term_vol_lookback=60,
        )
        assert strategy.config.base_q == 0.05
        assert strategy.config.observation_noise == 2.0

        _processed_df, signals = strategy.run(df)
        assert len(signals.entries) == len(df)

    def test_recommended_config(self) -> None:
        config = KalmanTrendStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["use_trailing_stop"] is True

    def test_get_startup_info(self) -> None:
        strategy = KalmanTrendStrategy()
        info = strategy.get_startup_info()
        assert isinstance(info, dict)
        assert "base_q" in info
        assert "observation_noise" in info
        assert "vel_threshold" in info
        assert "mode" in info

    def test_warmup_periods(self) -> None:
        strategy = KalmanTrendStrategy()
        # max(120, 20, 14) + 1 = 121
        assert strategy.warmup_periods() == 121

    def test_custom_config(self) -> None:
        config = KalmanTrendConfig(
            base_q=0.05,
            observation_noise=2.0,
            vol_lookback=10,
            long_term_vol_lookback=60,
        )
        strategy = KalmanTrendStrategy(config)
        assert strategy.config.base_q == 0.05

    def test_params_property(self) -> None:
        strategy = KalmanTrendStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "base_q" in params
        assert "observation_noise" in params
        assert "vel_threshold" in params

    def test_repr(self) -> None:
        strategy = KalmanTrendStrategy()
        assert "KalmanTrendStrategy" in repr(strategy)
