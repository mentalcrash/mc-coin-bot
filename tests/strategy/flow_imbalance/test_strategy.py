"""Tests for FlowImbalanceStrategy."""

import numpy as np
import pandas as pd

from src.strategy import get_strategy, list_strategies
from src.strategy.flow_imbalance import FlowImbalanceConfig, FlowImbalanceStrategy
from src.strategy.types import Direction


class TestRegistry:
    """Strategy Registry 테스트."""

    def test_registered(self):
        """'flow-imbalance'로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "flow-imbalance" in available

    def test_get_strategy(self):
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("flow-imbalance")
        assert strategy_class == FlowImbalanceStrategy


class TestFlowImbalanceStrategy:
    """FlowImbalanceStrategy 테스트."""

    def _make_sample_df(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.8)
        low = close - np.abs(np.random.randn(n) * 0.8)
        open_ = close + np.random.randn(n) * 0.3
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1h"),
        )

    def test_properties(self):
        """전략 속성 테스트."""
        strategy = FlowImbalanceStrategy()

        assert strategy.name == "Flow-Imbalance"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, FlowImbalanceConfig)

    def test_preprocess(self):
        """preprocess() 테스트."""
        df = self._make_sample_df()
        strategy = FlowImbalanceStrategy()
        processed = strategy.preprocess(df)

        expected_cols = [
            "returns",
            "buy_ratio",
            "buy_vol",
            "sell_vol",
            "ofi",
            "vpin_proxy",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self):
        """generate_signals() 테스트."""
        df = self._make_sample_df()
        strategy = FlowImbalanceStrategy()
        processed = strategy.preprocess(df)
        signals = strategy.generate_signals(processed)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_run_pipeline(self):
        """전체 파이프라인 (run) 테스트."""
        df = self._make_sample_df()
        strategy = FlowImbalanceStrategy()
        processed_df, signals = strategy.run(df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(df)
        assert len(signals.entries) == len(df)

    def test_from_params(self):
        """from_params()로 전략 생성."""
        df = self._make_sample_df()
        strategy = FlowImbalanceStrategy.from_params(
            ofi_window=12,
            vpin_window=48,
            timeout_bars=12,
        )

        assert strategy.config.ofi_window == 12
        assert strategy.config.timeout_bars == 12

        _processed_df, signals = strategy.run(df)
        assert len(signals.entries) == len(df)

    def test_recommended_config(self):
        """recommended_config() 테스트."""
        config = FlowImbalanceStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["use_trailing_stop"] is True

    def test_get_startup_info(self):
        """get_startup_info() 테스트."""
        strategy = FlowImbalanceStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "ofi_window" in info
        assert "vpin_threshold" in info
        assert "timeout" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        strategy = FlowImbalanceStrategy()
        warmup = strategy.warmup_periods()

        assert warmup == 25

    def test_custom_config(self):
        """커스텀 설정으로 전략 생성."""
        config = FlowImbalanceConfig(
            ofi_window=12,
            timeout_bars=48,
        )
        strategy = FlowImbalanceStrategy(config)

        assert strategy.config.ofi_window == 12
        assert strategy.config.timeout_bars == 48

    def test_params_property(self):
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = FlowImbalanceStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "ofi_window" in params
        assert "vpin_window" in params

    def test_repr(self):
        """문자열 표현 테스트."""
        strategy = FlowImbalanceStrategy()
        repr_str = repr(strategy)

        assert "FlowImbalanceStrategy" in repr_str
