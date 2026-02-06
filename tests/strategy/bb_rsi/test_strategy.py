"""Tests for BBRSIStrategy (Integration)."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.bb_rsi import BBRSIConfig, BBRSIStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """횡보 패턴의 샘플 OHLCV DataFrame."""
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
        """bb-rsi가 Registry에 등록됨."""
        available = list_strategies()
        assert "bb-rsi" in available

    def test_get_strategy(self):
        """get_strategy로 클래스 조회."""
        strategy_class = get_strategy("bb-rsi")
        assert strategy_class == BBRSIStrategy

    def test_other_strategies_still_registered(self):
        """기존 전략도 여전히 등록됨."""
        available = list_strategies()
        assert "tsmom" in available
        assert "donchian" in available


class TestBBRSIStrategy:
    """BBRSIStrategy 클래스 테스트."""

    def test_strategy_properties(self):
        """기본 속성 확인."""
        strategy = BBRSIStrategy()

        assert strategy.name == "BB-RSI"
        assert set(strategy.required_columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(strategy.config, BBRSIConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame):
        """run() end-to-end 파이프라인."""
        strategy = BBRSIStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 컬럼 확인
        assert "bb_upper" in processed_df.columns
        assert "rsi" in processed_df.columns
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
        strategy = BBRSIStrategy.from_params(
            bb_period=14,
            vol_target=0.30,
            bb_weight=0.5,
            rsi_weight=0.5,
        )

        assert strategy.config.bb_period == 14
        assert strategy.config.vol_target == 0.30

        # 실행 가능 확인
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self):
        """recommended_config 값 확인."""
        config = BBRSIStrategy.recommended_config()

        assert config["execution_mode"] == "orders"
        assert config["max_leverage_cap"] == 1.5
        assert config["use_trailing_stop"] is True

    def test_get_startup_info(self):
        """get_startup_info 키 확인."""
        strategy = BBRSIStrategy()
        info = strategy.get_startup_info()

        assert "bb_period" in info
        assert "rsi_period" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods 반환값."""
        strategy = BBRSIStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()

    def test_conservative_preset(self, sample_ohlcv: pd.DataFrame):
        """conservative() 팩토리 동작."""
        strategy = BBRSIStrategy.conservative()
        assert strategy.config.bb_period == 30
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_aggressive_preset(self, sample_ohlcv: pd.DataFrame):
        """aggressive() 팩토리 동작."""
        strategy = BBRSIStrategy.aggressive()
        assert strategy.config.bb_period == 14
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_for_timeframe(self):
        """for_timeframe() 팩토리."""
        strategy = BBRSIStrategy.for_timeframe("4h")
        assert strategy.config.annualization_factor == 2190.0

    def test_validate_input_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        strategy = BBRSIStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)

    def test_default_mode_is_long_only(self, sample_ohlcv: pd.DataFrame):
        """기본값은 Long-Only (ShortMode.DISABLED)."""
        strategy = BBRSIStrategy()
        _, signals = strategy.run(sample_ohlcv)

        assert (signals.direction >= 0).all()
        has_long = (signals.direction == Direction.LONG).any()
        assert has_long
