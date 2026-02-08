"""Tests for EnhancedTSMOMStrategy (Integration)."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.enhanced_tsmom import EnhancedTSMOMConfig, EnhancedTSMOMStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """트렌딩 패턴의 샘플 OHLCV DataFrame."""
    np.random.seed(42)
    n = 200

    base_price = 50000.0
    trend = np.linspace(0, 5000, n)
    noise = np.cumsum(np.random.randn(n) * 300)
    close = base_price + trend + noise

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
        """enhanced-tsmom이 Registry에 등록됨."""
        available = list_strategies()
        assert "enhanced-tsmom" in available

    def test_get_strategy(self):
        """get_strategy로 클래스 조회."""
        strategy_class = get_strategy("enhanced-tsmom")
        assert strategy_class == EnhancedTSMOMStrategy

    def test_other_strategies_still_registered(self):
        """기존 전략도 여전히 등록됨."""
        available = list_strategies()
        assert "tsmom" in available
        assert "bb-rsi" in available


class TestEnhancedTSMOMStrategy:
    """EnhancedTSMOMStrategy 클래스 테스트."""

    def test_strategy_properties(self):
        """기본 속성 확인."""
        strategy = EnhancedTSMOMStrategy()

        assert strategy.name == "Enhanced-VW-TSMOM"
        assert set(strategy.required_columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(strategy.config, EnhancedTSMOMConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame):
        """run() end-to-end 파이프라인."""
        strategy = EnhancedTSMOMStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 컬럼 확인
        assert "evw_momentum" in processed_df.columns
        assert "vol_scalar" in processed_df.columns
        assert "returns" in processed_df.columns
        assert "atr" in processed_df.columns

        # 시그널 구조 확인
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

        # direction 값 범위
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_from_params(self, sample_ohlcv: pd.DataFrame):
        """from_params()로 전략 생성 (parameter sweep 호환)."""
        strategy = EnhancedTSMOMStrategy.from_params(
            lookback=20,
            vol_target=0.40,
            volume_lookback=15,
            volume_clip_max=3.0,
        )

        assert strategy.config.lookback == 20
        assert strategy.config.vol_target == 0.40
        assert strategy.config.volume_lookback == 15
        assert strategy.config.volume_clip_max == 3.0

        # 실행 가능 확인
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self):
        """recommended_config 값 확인."""
        config = EnhancedTSMOMStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self):
        """get_startup_info 키 확인."""
        strategy = EnhancedTSMOMStrategy()
        info = strategy.get_startup_info()

        assert "lookback" in info
        assert "vol_target" in info
        assert "vol_window" in info
        assert "volume_lookback" in info
        assert "volume_clip_max" in info
        assert "mode" in info

    def test_warmup_periods(self):
        """warmup_periods 반환값."""
        strategy = EnhancedTSMOMStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()

    def test_conservative_preset(self, sample_ohlcv: pd.DataFrame):
        """conservative() 팩토리 동작."""
        strategy = EnhancedTSMOMStrategy.conservative()
        assert strategy.config.lookback == 48
        assert strategy.config.volume_lookback == 30
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_aggressive_preset(self, sample_ohlcv: pd.DataFrame):
        """aggressive() 팩토리 동작."""
        strategy = EnhancedTSMOMStrategy.aggressive()
        assert strategy.config.lookback == 12
        assert strategy.config.volume_lookback == 10
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_for_timeframe(self):
        """for_timeframe() 팩토리."""
        strategy = EnhancedTSMOMStrategy.for_timeframe("4h")
        assert strategy.config.annualization_factor == 2190.0

    def test_validate_input_missing_columns(self):
        """필수 컬럼 누락 시 에러."""
        strategy = EnhancedTSMOMStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)

    def test_default_mode_is_hedge_only(self, sample_ohlcv: pd.DataFrame):
        """기본값은 Hedge-Only (ShortMode.HEDGE_ONLY)."""
        strategy = EnhancedTSMOMStrategy()
        assert strategy.config.short_mode == 1  # HEDGE_ONLY

        _, signals = strategy.run(sample_ohlcv)
        has_long = (signals.direction == Direction.LONG).any()
        assert has_long
