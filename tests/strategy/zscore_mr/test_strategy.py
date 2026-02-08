"""Tests for ZScoreMRStrategy (Integration)."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.types import Direction
from src.strategy.zscore_mr import ZScoreMRConfig, ZScoreMRStrategy


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """평균회귀 패턴의 샘플 OHLCV DataFrame."""
    np.random.seed(42)
    n = 200

    base_price = 50000.0
    noise = np.cumsum(np.random.randn(n) * 300)
    close = base_price + noise - noise.mean()
    close = np.maximum(close, base_price * 0.8)

    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n) * 1000,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    """전략 Registry 통합 테스트."""

    def test_strategy_registered(self):
        """zscore-mr가 Registry에 등록됨."""
        available = list_strategies()
        assert "zscore-mr" in available

    def test_get_strategy(self):
        """get_strategy로 클래스 조회."""
        strategy_class = get_strategy("zscore-mr")
        assert strategy_class == ZScoreMRStrategy

    def test_other_strategies_still_registered(self):
        """기존 전략도 여전히 등록됨."""
        available = list_strategies()
        assert "tsmom" in available
        assert "bb-rsi" in available
        assert "donchian" in available


class TestZScoreMRStrategy:
    """ZScoreMRStrategy 클래스 테스트."""

    def test_strategy_properties(self):
        """기본 속성 확인."""
        strategy = ZScoreMRStrategy()

        assert strategy.name == "Z-Score MR"
        assert set(strategy.required_columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(strategy.config, ZScoreMRConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame):
        """run() end-to-end 파이프라인."""
        strategy = ZScoreMRStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 컬럼 확인
        assert "zscore" in processed_df.columns
        assert "vol_regime" in processed_df.columns
        assert "vol_scalar" in processed_df.columns

        # 시그널 구조 확인
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

        # direction 값 범위
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_from_params(self, sample_ohlcv: pd.DataFrame):
        """from_params()로 전략 생성 (parameter sweep 호환)."""
        strategy = ZScoreMRStrategy.from_params(
            short_lookback=15,
            long_lookback=50,
            entry_z=1.5,
            exit_z=0.3,
            vol_target=0.25,
        )

        assert strategy.config.short_lookback == 15
        assert strategy.config.long_lookback == 50
        assert strategy.config.entry_z == 1.5
        assert strategy.config.vol_target == 0.25

        # 실행 가능 확인
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self):
        """recommended_config 값 확인."""
        config = ZScoreMRStrategy.recommended_config()

        assert config["execution_mode"] == "orders"
        assert config["max_leverage_cap"] == 1.5
        assert config["rebalance_threshold"] == 0.03
        assert config["use_trailing_stop"] is True

    def test_get_startup_info(self):
        """get_startup_info 키 확인."""
        strategy = ZScoreMRStrategy()
        info = strategy.get_startup_info()

        assert "lookback" in info
        assert "entry/exit_z" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods 반환값."""
        strategy = ZScoreMRStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()

    def test_for_timeframe(self):
        """for_timeframe() 팩토리."""
        strategy = ZScoreMRStrategy.for_timeframe("4h")
        assert strategy.config.annualization_factor == 2190.0

    def test_validate_input_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        strategy = ZScoreMRStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)

    def test_default_mode_is_full(self, sample_ohlcv: pd.DataFrame):
        """기본값은 Long/Short (ShortMode.FULL) - 평균회귀 전략 특성."""
        strategy = ZScoreMRStrategy()
        _, signals = strategy.run(sample_ohlcv)

        has_long = (signals.direction == Direction.LONG).any()
        has_short = (signals.direction == Direction.SHORT).any()
        assert has_long
        assert has_short

    def test_params_property(self):
        """params 프로퍼티가 config를 dict로 반환."""
        strategy = ZScoreMRStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "short_lookback" in params
        assert "entry_z" in params

    def test_repr(self):
        """repr 문자열."""
        strategy = ZScoreMRStrategy()
        repr_str = repr(strategy)
        assert "ZScoreMRStrategy" in repr_str
