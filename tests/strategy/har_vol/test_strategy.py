"""Tests for HARVolStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.har_vol import HARVolConfig, HARVolStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (400일)."""
    np.random.seed(42)
    n = 400

    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5) + 0.01
    low = close - np.abs(np.random.randn(n) * 1.5) - 0.01
    open_ = close + np.random.randn(n) * 0.5

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )

    return df


class TestRegistry:
    """Strategy Registry 테스트."""

    def test_registered(self):
        """'har-vol'로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "har-vol" in available

    def test_get_strategy(self):
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("har-vol")
        assert strategy_class == HARVolStrategy


class TestHARVolStrategy:
    """HARVolStrategy 테스트."""

    def test_properties(self):
        """전략 속성 테스트."""
        strategy = HARVolStrategy()

        assert strategy.name == "HAR Volatility"
        assert set(strategy.required_columns) == {
            "close",
            "high",
            "low",
            "volume",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, HARVolConfig)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame):
        """전체 파이프라인 (run) 테스트."""
        config = HARVolConfig(
            daily_window=1,
            weekly_window=3,
            monthly_window=15,
            training_window=60,
        )
        strategy = HARVolStrategy(config)
        processed_df, signals = strategy.run(sample_ohlcv_df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

        # direction 값 검증
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_from_params(self, sample_ohlcv_df: pd.DataFrame):
        """from_params()로 전략 생성."""
        strategy = HARVolStrategy.from_params(
            daily_window=2,
            weekly_window=5,
            monthly_window=20,
            training_window=100,
        )

        assert strategy.config.daily_window == 2
        assert strategy.config.weekly_window == 5
        assert strategy.config.monthly_window == 20
        assert strategy.config.training_window == 100

        # 전체 파이프라인 정상 동작 확인
        _processed_df, signals = strategy.run(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_recommended_config(self):
        """recommended_config() 테스트."""
        config = HARVolStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_startup_info(self):
        """get_startup_info() 테스트."""
        strategy = HARVolStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "daily_window" in info
        assert "weekly_window" in info
        assert "monthly_window" in info
        assert "training_window" in info
        assert "vol_surprise_threshold" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_warmup(self):
        """warmup_periods() 테스트."""
        strategy = HARVolStrategy()
        warmup = strategy.warmup_periods()

        # 252 + 22 + 1 = 275
        assert warmup == 275

    def test_validate_input(self, sample_ohlcv_df: pd.DataFrame):
        """validate_input() 테스트."""
        strategy = HARVolStrategy()

        # 정상 입력
        strategy.validate_input(sample_ohlcv_df)

        # 빈 DataFrame
        with pytest.raises(ValueError, match="empty"):
            strategy.validate_input(pd.DataFrame())

        # 누락 컬럼
        df_missing = sample_ohlcv_df.drop(columns=["high"])
        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.validate_input(df_missing)

    def test_params_property(self):
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = HARVolStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "daily_window" in params
        assert "weekly_window" in params
        assert "monthly_window" in params
        assert "training_window" in params
        assert "vol_surprise_threshold" in params

    def test_repr(self):
        """문자열 표현 테스트."""
        strategy = HARVolStrategy()
        repr_str = repr(strategy)

        assert "HARVolStrategy" in repr_str
