"""Tests for GKBreakoutStrategy (Integration)."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.gk_breakout import GKBreakoutConfig, GKBreakoutStrategy


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """추세 + 횡보 패턴의 샘플 OHLCV DataFrame."""
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
        """gk-breakout이 Registry에 등록됨."""
        available = list_strategies()
        assert "gk-breakout" in available

    def test_get_strategy(self):
        """get_strategy로 클래스 조회."""
        strategy_class = get_strategy("gk-breakout")
        assert strategy_class == GKBreakoutStrategy

    def test_other_strategies_still_registered(self):
        """기존 전략도 여전히 등록됨."""
        available = list_strategies()
        assert "tsmom" in available
        assert "donchian" in available
        assert "bb-rsi" in available


class TestGKBreakoutStrategy:
    """GKBreakoutStrategy 클래스 테스트."""

    def test_strategy_properties(self):
        """기본 속성 확인."""
        strategy = GKBreakoutStrategy()

        assert strategy.name == "GK-Breakout"
        assert set(strategy.required_columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(strategy.config, GKBreakoutConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame):
        """run() end-to-end 파이프라인."""
        strategy = GKBreakoutStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 컬럼 확인
        assert "gk_var" in processed_df.columns
        assert "vol_ratio" in processed_df.columns
        assert "dc_upper" in processed_df.columns
        assert "dc_lower" in processed_df.columns
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
        strategy = GKBreakoutStrategy.from_params(
            gk_lookback=15,
            compression_threshold=0.80,
            breakout_lookback=15,
            vol_target=0.25,
        )

        assert strategy.config.gk_lookback == 15
        assert strategy.config.compression_threshold == 0.80
        assert strategy.config.breakout_lookback == 15
        assert strategy.config.vol_target == 0.25

        # 실행 가능 확인
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self):
        """recommended_config 값 확인."""
        config = GKBreakoutStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05

    def test_get_startup_info(self):
        """get_startup_info 키 확인."""
        strategy = GKBreakoutStrategy()
        info = strategy.get_startup_info()

        assert "gk_lookback" in info
        assert "compression_threshold" in info
        assert "breakout_lookback" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods 반환값."""
        strategy = GKBreakoutStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()

    def test_validate_input_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        strategy = GKBreakoutStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)

    def test_default_mode_is_hedge_only(self, sample_ohlcv: pd.DataFrame):
        """기본값은 HEDGE_ONLY."""
        strategy = GKBreakoutStrategy()
        assert strategy.config.short_mode == 1  # ShortMode.HEDGE_ONLY

    def test_params_property(self):
        """params 프로퍼티가 config 딕셔너리를 반환."""
        strategy = GKBreakoutStrategy()
        params = strategy.params
        assert isinstance(params, dict)
        assert "gk_lookback" in params
        assert "compression_threshold" in params
        assert "breakout_lookback" in params

    def test_repr(self):
        """__repr__ 문자열 표현."""
        strategy = GKBreakoutStrategy()
        repr_str = repr(strategy)
        assert "GKBreakoutStrategy" in repr_str
