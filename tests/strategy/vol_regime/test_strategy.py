"""Tests for VolRegimeStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.types import Direction
from src.strategy.vol_regime import VolRegimeConfig, VolRegimeStrategy


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (350일)."""
    np.random.seed(42)
    n = 350

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

    def test_registered_as_vol_regime(self):
        """'vol-regime'으로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "vol-regime" in available

    def test_get_strategy(self):
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("vol-regime")
        assert strategy_class == VolRegimeStrategy


class TestVolRegimeStrategy:
    """VolRegimeStrategy 테스트."""

    def test_strategy_properties(self):
        """전략 속성 테스트."""
        strategy = VolRegimeStrategy()

        assert strategy.name == "Vol-Regime"
        assert set(strategy.required_columns) == {
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, VolRegimeConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame):
        """preprocess() 테스트."""
        strategy = VolRegimeStrategy()
        processed = strategy.preprocess(sample_ohlcv_df)

        expected_cols = [
            "returns",
            "realized_vol",
            "vol_regime",
            "regime_strength",
            "atr",
            "drawdown",
        ]
        for col in expected_cols:
            assert col in processed.columns

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame):
        """generate_signals() 테스트."""
        strategy = VolRegimeStrategy()
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
        strategy = VolRegimeStrategy()
        processed_df, signals = strategy.run(sample_ohlcv_df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_from_params(self, sample_ohlcv_df: pd.DataFrame):
        """from_params()로 전략 생성."""
        strategy = VolRegimeStrategy.from_params(
            vol_lookback=30,
            high_vol_threshold=0.75,
            low_vol_threshold=0.25,
        )

        assert strategy.config.vol_lookback == 30
        assert strategy.config.high_vol_threshold == 0.75
        assert strategy.config.low_vol_threshold == 0.25

        # 전체 파이프라인 정상 동작 확인
        _processed_df, signals = strategy.run(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_recommended_config(self):
        """recommended_config() 테스트."""
        config = VolRegimeStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05

    def test_get_startup_info(self):
        """get_startup_info() 테스트."""
        strategy = VolRegimeStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "vol_lookback" in info
        assert "vol_rank_lookback" in info
        assert "high_vol" in info
        assert "normal" in info
        assert "low_vol" in info
        assert "thresholds" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        strategy = VolRegimeStrategy()
        warmup = strategy.warmup_periods()

        # max(252, 60, 30, 14) + 1 = 253
        assert warmup == 253

    def test_custom_config(self):
        """커스텀 설정으로 전략 생성."""
        config = VolRegimeConfig(
            vol_lookback=30,
            high_vol_lookback=90,
            normal_lookback=45,
            low_vol_lookback=20,
        )
        strategy = VolRegimeStrategy(config)

        assert strategy.config.vol_lookback == 30
        assert strategy.config.high_vol_lookback == 90

    def test_params_property(self):
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = VolRegimeStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "vol_lookback" in params
        assert "vol_rank_lookback" in params
        assert "high_vol_threshold" in params

    def test_repr(self):
        """문자열 표현 테스트."""
        strategy = VolRegimeStrategy()
        repr_str = repr(strategy)

        assert "VolRegimeStrategy" in repr_str
