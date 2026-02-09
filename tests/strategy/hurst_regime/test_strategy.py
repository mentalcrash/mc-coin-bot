"""Tests for HurstRegimeStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.hurst_regime import HurstRegimeConfig, HurstRegimeStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (250일)."""
    np.random.seed(42)
    n = 250

    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5

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


class TestRegistry:
    """Strategy Registry 테스트."""

    def test_registered(self):
        """'hurst-regime'으로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "hurst-regime" in available

    def test_get_strategy(self):
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("hurst-regime")
        assert strategy_class == HurstRegimeStrategy


class TestHurstRegimeStrategy:
    """HurstRegimeStrategy 테스트."""

    def test_properties(self):
        """전략 속성 테스트."""
        strategy = HurstRegimeStrategy()

        assert strategy.name == "Hurst-Regime"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, HurstRegimeConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame):
        """preprocess() 테스트."""
        strategy = HurstRegimeStrategy()
        processed = strategy.preprocess(sample_ohlcv_df)

        expected_cols = [
            "returns",
            "er",
            "hurst",
            "momentum",
            "z_score",
            "realized_vol",
            "vol_scalar",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns, f"Missing column: {col}"

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame):
        """generate_signals() 테스트."""
        strategy = HurstRegimeStrategy()
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

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame):
        """전체 파이프라인 (run) 테스트."""
        strategy = HurstRegimeStrategy()
        processed_df, signals = strategy.run(sample_ohlcv_df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self, sample_ohlcv_df: pd.DataFrame):
        """from_params()로 전략 생성."""
        strategy = HurstRegimeStrategy.from_params(
            er_lookback=30,
            hurst_window=80,
            er_trend_threshold=0.7,
            er_mr_threshold=0.2,
        )

        assert strategy.config.er_lookback == 30
        assert strategy.config.hurst_window == 80
        assert strategy.config.er_trend_threshold == 0.7
        assert strategy.config.er_mr_threshold == 0.2

        # 전체 파이프라인 정상 동작 확인
        _processed_df, signals = strategy.run(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_recommended_config(self):
        """recommended_config() 테스트."""
        config = HurstRegimeStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self):
        """get_startup_info() 테스트."""
        strategy = HurstRegimeStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "er_lookback" in info
        assert "hurst_window" in info
        assert "er_thresholds" in info
        assert "hurst_thresholds" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        strategy = HurstRegimeStrategy()
        warmup = strategy.warmup_periods()

        # max(100, 20, 20, 20, 20, 14) + 1 = 101
        assert warmup == 101

    def test_custom_config(self):
        """커스텀 설정으로 전략 생성."""
        config = HurstRegimeConfig(
            er_lookback=30,
            hurst_window=80,
            mom_lookback=15,
            mr_lookback=25,
        )
        strategy = HurstRegimeStrategy(config)

        assert strategy.config.er_lookback == 30
        assert strategy.config.hurst_window == 80
        assert strategy.config.mom_lookback == 15
        assert strategy.config.mr_lookback == 25

    def test_params_property(self):
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = HurstRegimeStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "er_lookback" in params
        assert "hurst_window" in params
        assert "er_trend_threshold" in params
        assert "er_mr_threshold" in params
        assert "hurst_trend_threshold" in params
        assert "hurst_mr_threshold" in params

    def test_repr(self):
        """문자열 표현 테스트."""
        strategy = HurstRegimeStrategy()
        repr_str = repr(strategy)

        assert "HurstRegimeStrategy" in repr_str
