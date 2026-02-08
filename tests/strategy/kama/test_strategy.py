"""Tests for KAMAStrategy (Integration)."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.kama import KAMAConfig, KAMAStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """상승 추세 패턴의 샘플 OHLCV DataFrame."""
    np.random.seed(42)
    n = 200

    base_price = 50000.0
    trend = np.linspace(0, 5000, n)
    noise = np.cumsum(np.random.randn(n) * 300)
    close = base_price + trend + noise
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
        """kama가 Registry에 등록됨."""
        available = list_strategies()
        assert "kama" in available

    def test_get_strategy(self):
        """get_strategy로 클래스 조회."""
        strategy_class = get_strategy("kama")
        assert strategy_class == KAMAStrategy

    def test_other_strategies_still_registered(self):
        """기존 전략도 여전히 등록됨."""
        available = list_strategies()
        assert "tsmom" in available
        assert "bb-rsi" in available
        assert "donchian" in available


class TestKAMAStrategy:
    """KAMAStrategy 클래스 테스트."""

    def test_strategy_properties(self):
        """기본 속성 확인."""
        strategy = KAMAStrategy()

        assert strategy.name == "KAMA"
        assert set(strategy.required_columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(strategy.config, KAMAConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame):
        """run() end-to-end 파이프라인."""
        strategy = KAMAStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 컬럼 확인
        assert "kama" in processed_df.columns
        assert "atr" in processed_df.columns
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
        strategy = KAMAStrategy.from_params(
            er_lookback=15,
            fast_period=3,
            slow_period=40,
            vol_target=0.25,
        )

        assert strategy.config.er_lookback == 15
        assert strategy.config.fast_period == 3
        assert strategy.config.slow_period == 40
        assert strategy.config.vol_target == 0.25

        # 실행 가능 확인
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self):
        """recommended_config 값 확인."""
        config = KAMAStrategy.recommended_config()

        assert config["execution_mode"] == "orders"
        assert config["max_leverage_cap"] == 2.0
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self):
        """get_startup_info 키 확인."""
        strategy = KAMAStrategy()
        info = strategy.get_startup_info()

        assert "er_lookback" in info
        assert "fast/slow" in info
        assert "atr_multiplier" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods 반환값."""
        strategy = KAMAStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()

    def test_validate_input_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        strategy = KAMAStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)

    def test_default_mode_is_hedge_only(self, sample_ohlcv: pd.DataFrame):
        """기본값은 HEDGE_ONLY."""
        strategy = KAMAStrategy()
        assert strategy.config.short_mode == Direction.LONG  # ShortMode.HEDGE_ONLY == 1
        _, signals = strategy.run(sample_ohlcv)

        # HEDGE_ONLY에서 숏 시그널은 드로다운 시에만 가능
        # direction 값이 유효 범위에 있음을 검증
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})
