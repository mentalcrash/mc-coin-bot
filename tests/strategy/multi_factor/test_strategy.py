"""Tests for MultiFactorStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.multi_factor import MultiFactorConfig, MultiFactorStrategy
from src.strategy.types import Direction


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
        """'multi-factor'로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "multi-factor" in available

    def test_get_strategy(self) -> None:
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("multi-factor")
        assert strategy_class == MultiFactorStrategy


class TestMultiFactorStrategy:
    """MultiFactorStrategy 테스트."""

    def test_properties(self) -> None:
        """전략 속성 테스트."""
        strategy = MultiFactorStrategy()

        assert strategy.name == "Multi-Factor Ensemble"
        assert set(strategy.required_columns) == {
            "close",
            "high",
            "low",
            "volume",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, MultiFactorConfig)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """전체 파이프라인 (run) 테스트."""
        strategy = MultiFactorStrategy()
        processed_df, signals = strategy.run(sample_ohlcv_df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

        # 시그널 타입 검증
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_from_params(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """from_params()로 전략 생성."""
        strategy = MultiFactorStrategy.from_params(
            momentum_lookback=30,
            volume_shock_window=10,
            vol_target=0.40,
        )

        assert strategy.config.momentum_lookback == 30
        assert strategy.config.volume_shock_window == 10
        assert strategy.config.vol_target == 0.40

        # 전체 파이프라인 정상 동작 확인
        _processed_df, signals = strategy.run(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_recommended_config(self) -> None:
        """recommended_config() 테스트."""
        config = MultiFactorStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self) -> None:
        """get_startup_info() 테스트."""
        strategy = MultiFactorStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "momentum_lookback" in info
        assert "volume_shock_window" in info
        assert "vol_window" in info
        assert "vol_target" in info
        assert "zscore_window" in info
        assert "mode" in info

    def test_warmup_periods(self) -> None:
        """warmup_periods() 테스트."""
        strategy = MultiFactorStrategy()
        warmup = strategy.warmup_periods()

        # max(21, 30, 60) + 1 = 61
        assert warmup == 61

    def test_custom_config(self) -> None:
        """커스텀 설정으로 전략 생성."""
        config = MultiFactorConfig(
            momentum_lookback=30,
            volume_shock_window=10,
            vol_window=40,
            zscore_window=100,
        )
        strategy = MultiFactorStrategy(config)

        assert strategy.config.momentum_lookback == 30
        assert strategy.config.volume_shock_window == 10
        assert strategy.config.vol_window == 40
        assert strategy.config.zscore_window == 100

    def test_validate_input_missing_columns(self) -> None:
        """필수 컬럼 누락 시 validate_input 에러."""
        strategy = MultiFactorStrategy()
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.run(df)

    def test_params_property(self) -> None:
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = MultiFactorStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "momentum_lookback" in params
        assert "volume_shock_window" in params
        assert "vol_window" in params
        assert "vol_target" in params
        assert "zscore_window" in params
        assert "short_mode" in params

    def test_repr(self) -> None:
        """문자열 표현 테스트."""
        strategy = MultiFactorStrategy()
        repr_str = repr(strategy)

        assert "MultiFactorStrategy" in repr_str
