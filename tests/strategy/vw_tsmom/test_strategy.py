"""Tests for VWTSMOMStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.types import Direction
from src.strategy.vw_tsmom import VWTSMOMConfig, VWTSMOMStrategy


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

    def test_strategy_registered(self):
        """'vw-tsmom'으로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "vw-tsmom" in available

    def test_get_strategy(self):
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("vw-tsmom")
        assert strategy_class == VWTSMOMStrategy


class TestVWTSMOMStrategy:
    """VWTSMOMStrategy 테스트."""

    def test_strategy_properties(self):
        """전략 속성 테스트."""
        strategy = VWTSMOMStrategy()

        assert strategy.name == "VW-TSMOM Pure"
        assert set(strategy.required_columns) == {"close", "volume"}
        assert strategy.config is not None
        assert isinstance(strategy.config, VWTSMOMConfig)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame):
        """전체 파이프라인 (run) 테스트."""
        strategy = VWTSMOMStrategy()
        processed_df, signals = strategy.run(sample_ohlcv_df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_from_params(self, sample_ohlcv_df: pd.DataFrame):
        """from_params()로 전략 생성."""
        strategy = VWTSMOMStrategy.from_params(
            lookback=15,
            vol_window=20,
            vol_target=0.40,
        )

        assert strategy.config.lookback == 15
        assert strategy.config.vol_window == 20
        assert strategy.config.vol_target == 0.40

        # 전체 파이프라인 정상 동작 확인
        _processed_df, signals = strategy.run(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_recommended_config(self):
        """recommended_config() 테스트."""
        config = VWTSMOMStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self):
        """get_startup_info() 테스트."""
        strategy = VWTSMOMStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "lookback" in info
        assert "vol_target" in info
        assert "vol_window" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods() 테스트."""
        strategy = VWTSMOMStrategy()
        warmup = strategy.warmup_periods()

        # max(21, 30) + 1 = 31
        assert warmup == 31

    def test_validate_input_missing_columns(self):
        """필수 컬럼 누락 시 validate_input에서 에러."""
        strategy = VWTSMOMStrategy()

        # 'volume' 컬럼 없음
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.validate_input(df)

    def test_params_property(self):
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = VWTSMOMStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "lookback" in params
        assert "vol_window" in params
        assert "vol_target" in params
        assert "short_mode" in params

    def test_repr(self):
        """문자열 표현 테스트."""
        strategy = VWTSMOMStrategy()
        repr_str = repr(strategy)

        assert "VWTSMOMStrategy" in repr_str
