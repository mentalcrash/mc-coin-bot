"""Tests for RangeSqueezeStrategy."""

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.range_squeeze import RangeSqueezeConfig, RangeSqueezeStrategy
from src.strategy.types import Direction


class TestRegistry:
    """Strategy Registry 테스트."""

    def test_registered(self):
        """'range-squeeze'로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "range-squeeze" in available

    def test_get_strategy(self):
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("range-squeeze")
        assert strategy_class == RangeSqueezeStrategy


class TestRangeSqueezeStrategy:
    """RangeSqueezeStrategy 테스트."""

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
        """전략 속성 테스트."""
        strategy = RangeSqueezeStrategy()

        assert strategy.name == "Range-Squeeze"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, RangeSqueezeConfig)

    def test_preprocess(self):
        """preprocess() 테스트."""
        df = self._make_sample_df()
        strategy = RangeSqueezeStrategy()
        processed = strategy.preprocess(df)

        expected_cols = [
            "returns",
            "daily_range",
            "avg_range",
            "range_ratio",
            "is_nr",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self):
        """generate_signals() 테스트."""
        df = self._make_sample_df()
        strategy = RangeSqueezeStrategy()
        processed = strategy.preprocess(df)
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

    def test_run_pipeline(self):
        """전체 파이프라인 (run) 테스트."""
        df = self._make_sample_df()
        strategy = RangeSqueezeStrategy()
        processed_df, signals = strategy.run(df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self):
        """from_params()로 전략 생성."""
        df = self._make_sample_df()
        strategy = RangeSqueezeStrategy.from_params(
            nr_period=10,
            lookback=30,
            squeeze_threshold=0.6,
        )

        assert strategy.config.nr_period == 10
        assert strategy.config.lookback == 30
        assert strategy.config.squeeze_threshold == 0.6

        _processed_df, signals = strategy.run(df)
        assert len(signals.entries) == len(df)

    def test_recommended_config(self):
        """recommended_config() 테스트."""
        config = RangeSqueezeStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self):
        """get_startup_info() 테스트."""
        strategy = RangeSqueezeStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "nr_period" in info
        assert "lookback" in info
        assert "squeeze_threshold" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        strategy = RangeSqueezeStrategy()
        warmup = strategy.warmup_periods()

        # max(20, 7, 14) + 1 = 21
        assert warmup == 21

    def test_custom_config(self):
        """커스텀 설정으로 전략 생성."""
        config = RangeSqueezeConfig(
            nr_period=10,
            lookback=40,
        )
        strategy = RangeSqueezeStrategy(config)

        assert strategy.config.nr_period == 10
        assert strategy.config.lookback == 40

    def test_params_property(self):
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = RangeSqueezeStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "nr_period" in params
        assert "lookback" in params
        assert "squeeze_threshold" in params

    def test_repr(self):
        """문자열 표현 테스트."""
        strategy = RangeSqueezeStrategy()
        repr_str = repr(strategy)

        assert "RangeSqueezeStrategy" in repr_str
