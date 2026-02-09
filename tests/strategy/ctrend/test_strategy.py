"""Tests for CTRENDStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.ctrend import CTRENDConfig, CTRENDStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (400일, training_window=252 고려)."""
    np.random.seed(42)
    n = 400

    trend = np.linspace(0, 60, n)
    noise = np.cumsum(np.random.randn(n) * 2)
    close = 100 + trend + noise
    high = close + np.abs(np.random.randn(n) * 1.5) + 0.5
    low = close - np.abs(np.random.randn(n) * 1.5) - 0.5
    open_ = close + np.random.randn(n) * 0.5

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    """Strategy Registry 테스트."""

    def test_registered(self) -> None:
        """'ctrend'로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "ctrend" in available

    def test_get_strategy(self) -> None:
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("ctrend")
        assert strategy_class == CTRENDStrategy


class TestCTRENDStrategy:
    """CTRENDStrategy 테스트."""

    def test_properties(self) -> None:
        """전략 속성 테스트."""
        strategy = CTRENDStrategy()

        assert strategy.name == "CTREND"
        assert set(strategy.required_columns) == {
            "close",
            "high",
            "low",
            "volume",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, CTRENDConfig)

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """전체 파이프라인 (run) 테스트."""
        # training_window=100 for faster test
        config = CTRENDConfig(training_window=100)
        strategy = CTRENDStrategy(config)
        processed_df, signals = strategy.run(sample_ohlcv_df)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

        # direction은 유효한 값만
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_from_params(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """from_params()로 전략 생성."""
        strategy = CTRENDStrategy.from_params(
            training_window=100,
            prediction_horizon=3,
            alpha=0.7,
        )

        assert strategy.config.training_window == 100
        assert strategy.config.prediction_horizon == 3
        assert strategy.config.alpha == 0.7

        # 전체 파이프라인 정상 동작 확인
        _processed_df, signals = strategy.run(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_recommended_config(self) -> None:
        """recommended_config() 테스트."""
        config = CTRENDStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self) -> None:
        """get_startup_info() 테스트."""
        strategy = CTRENDStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "training_window" in info
        assert "prediction_horizon" in info
        assert "alpha" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_warmup_periods(self) -> None:
        """warmup_periods() 테스트."""
        strategy = CTRENDStrategy()
        warmup = strategy.warmup_periods()

        # training_window(252) + 50 = 302
        assert warmup == 302

    def test_validate_input(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """입력 검증 테스트."""
        strategy = CTRENDStrategy()

        # 정상 입력은 에러 없음
        strategy.validate_input(sample_ohlcv_df)

        # 빈 DataFrame은 에러
        with pytest.raises(ValueError, match="empty"):
            strategy.validate_input(pd.DataFrame())

        # 필수 컬럼 누락
        with pytest.raises(ValueError, match="Missing"):
            strategy.validate_input(
                pd.DataFrame(
                    {"close": [100]},
                    index=pd.DatetimeIndex(["2024-01-01"]),
                )
            )

    def test_custom_config(self) -> None:
        """커스텀 설정으로 전략 생성."""
        config = CTRENDConfig(
            training_window=120,
            prediction_horizon=10,
            alpha=0.8,
            vol_target=0.25,
        )
        strategy = CTRENDStrategy(config)

        assert strategy.config.training_window == 120
        assert strategy.config.prediction_horizon == 10
        assert strategy.config.alpha == 0.8
        assert strategy.config.vol_target == 0.25

    def test_params_property(self) -> None:
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = CTRENDStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "training_window" in params
        assert "prediction_horizon" in params
        assert "alpha" in params
        assert "vol_target" in params
        assert "short_mode" in params
