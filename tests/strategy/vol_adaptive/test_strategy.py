"""Tests for VolAdaptiveStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.types import Direction
from src.strategy.vol_adaptive import VolAdaptiveConfig, VolAdaptiveStrategy


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (200일)."""
    np.random.seed(42)
    n = 200

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

    def test_registered(self) -> None:
        """'vol-adaptive'로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "vol-adaptive" in available

    def test_get_strategy(self) -> None:
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("vol-adaptive")
        assert strategy_class == VolAdaptiveStrategy


class TestVolAdaptiveStrategy:
    """VolAdaptiveStrategy 테스트."""

    def test_properties(self) -> None:
        """전략 속성 테스트."""
        strategy = VolAdaptiveStrategy()

        assert strategy.name == "Vol-Adaptive"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, VolAdaptiveConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """preprocess() 테스트."""
        strategy = VolAdaptiveStrategy()
        processed = strategy.preprocess(sample_ohlcv_df)

        expected_cols = [
            "returns",
            "realized_vol",
            "vol_scalar",
            "ema_fast",
            "ema_slow",
            "rsi",
            "adx",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """generate_signals() 테스트."""
        strategy = VolAdaptiveStrategy()
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

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """전체 파이프라인 (run) 테스트."""
        strategy = VolAdaptiveStrategy()
        processed_df, signals = strategy.run(sample_ohlcv_df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """from_params()로 전략 생성."""
        strategy = VolAdaptiveStrategy.from_params(
            ema_fast=8,
            ema_slow=40,
            adx_threshold=25.0,
        )

        assert strategy.config.ema_fast == 8
        assert strategy.config.ema_slow == 40
        assert strategy.config.adx_threshold == 25.0

        # 전체 파이프라인 정상 동작 확인
        _processed_df, signals = strategy.run(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_recommended_config(self) -> None:
        """recommended_config() 테스트."""
        config = VolAdaptiveStrategy.recommended_config()

        assert config["execution_mode"] == "orders"
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self) -> None:
        """get_startup_info() 테스트."""
        strategy = VolAdaptiveStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "ema" in info
        assert "rsi" in info
        assert "adx" in info
        assert "vol_target" in info
        assert "atr_period" in info
        assert "mode" in info

    def test_warmup_periods(self) -> None:
        """warmup_periods() 테스트."""
        strategy = VolAdaptiveStrategy()
        warmup = strategy.warmup_periods()

        # max(50, 20, 14, 14, 14) + 1 = 51
        assert warmup == 51

    def test_custom_config(self) -> None:
        """커스텀 설정으로 전략 생성."""
        config = VolAdaptiveConfig(
            ema_fast=8,
            ema_slow=40,
            rsi_period=10,
            adx_period=20,
        )
        strategy = VolAdaptiveStrategy(config)

        assert strategy.config.ema_fast == 8
        assert strategy.config.ema_slow == 40
        assert strategy.config.rsi_period == 10
        assert strategy.config.adx_period == 20

    def test_params_property(self) -> None:
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = VolAdaptiveStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "ema_fast" in params
        assert "ema_slow" in params
        assert "rsi_period" in params
        assert "adx_period" in params
        assert "adx_threshold" in params
        assert "vol_target" in params

    def test_repr(self) -> None:
        """문자열 표현 테스트."""
        strategy = VolAdaptiveStrategy()
        repr_str = repr(strategy)

        assert "VolAdaptiveStrategy" in repr_str
