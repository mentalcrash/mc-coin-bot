"""Tests for AdaptiveBreakoutStrategy."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.breakout import AdaptiveBreakoutConfig, AdaptiveBreakoutStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성."""
    np.random.seed(42)
    n = 100

    # 상승 추세 + 노이즈
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
            "volume": np.random.randint(1000, 10000, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )

    return df


class TestAdaptiveBreakoutStrategy:
    """AdaptiveBreakoutStrategy 테스트."""

    def test_strategy_registration(self):
        """전략이 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "adaptive-breakout" in available
        assert "tsmom" in available

    def test_get_strategy(self):
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("adaptive-breakout")
        assert strategy_class == AdaptiveBreakoutStrategy

    def test_strategy_properties(self):
        """전략 속성 테스트."""
        strategy = AdaptiveBreakoutStrategy()

        assert strategy.name == "AdaptiveBreakout"
        assert set(strategy.required_columns) == {"open", "high", "low", "close"}
        assert strategy.config is not None
        assert isinstance(strategy.config, AdaptiveBreakoutConfig)

    def test_preprocess(self, sample_ohlcv_df: pd.DataFrame):
        """preprocess() 테스트."""
        strategy = AdaptiveBreakoutStrategy()
        processed = strategy.preprocess(sample_ohlcv_df)

        # 계산된 컬럼 확인
        assert "upper_band" in processed.columns
        assert "lower_band" in processed.columns
        assert "middle_band" in processed.columns
        assert "atr" in processed.columns
        assert "realized_vol" in processed.columns
        assert "vol_scalar" in processed.columns
        assert "threshold" in processed.columns

        # NaN 확인 (워밍업 기간 이후)
        warmup = strategy.config.warmup_periods()
        assert not processed["upper_band"].iloc[warmup:].isna().any()
        assert not processed["atr"].iloc[warmup:].isna().any()

    def test_generate_signals(self, sample_ohlcv_df: pd.DataFrame):
        """generate_signals() 테스트."""
        strategy = AdaptiveBreakoutStrategy()
        processed = strategy.preprocess(sample_ohlcv_df)
        signals = strategy.generate_signals(processed)

        # 시그널 구조 확인
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

        # 시그널 타입 확인
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

        # 방향 값 범위 확인
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_run_pipeline(self, sample_ohlcv_df: pd.DataFrame):
        """전체 파이프라인 (run) 테스트."""
        strategy = AdaptiveBreakoutStrategy()
        processed_df, signals = strategy.run(sample_ohlcv_df)

        # 반환값 확인
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_df)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_recommended_portfolio(self):
        """recommended_portfolio() 테스트."""
        # NOTE: Portfolio 임포트 시 순환 참조 가능성 있으므로 지연 임포트
        try:
            portfolio = AdaptiveBreakoutStrategy.recommended_portfolio(initial_capital=50000)

            assert portfolio.initial_capital == Decimal("50000")
            assert portfolio.config.max_leverage_cap == 2.5
            assert portfolio.config.system_stop_loss == 0.08
            assert portfolio.config.rebalance_threshold == 0.03
        except ImportError as e:
            pytest.skip(f"Skipping due to circular import: {e}")

    def test_get_startup_info(self):
        """get_startup_info() 테스트."""
        strategy = AdaptiveBreakoutStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "channel_period" in info
        assert "k_value" in info
        assert "atr_period" in info
        assert "mode" in info

    def test_factory_conservative(self):
        """conservative() 팩토리 테스트."""
        strategy = AdaptiveBreakoutStrategy.conservative()

        assert strategy.config.channel_period == 30
        assert strategy.config.k_value == 2.0

    def test_factory_aggressive(self):
        """aggressive() 팩토리 테스트."""
        strategy = AdaptiveBreakoutStrategy.aggressive()

        assert strategy.config.channel_period == 10
        assert strategy.config.k_value == 1.0

    def test_long_only_mode(self, sample_ohlcv_df: pd.DataFrame):
        """Long-Only 모드 테스트."""
        config = AdaptiveBreakoutConfig(long_only=True)
        strategy = AdaptiveBreakoutStrategy(config)

        _processed_df, signals = strategy.run(sample_ohlcv_df)

        # Short 시그널이 없어야 함
        assert (signals.direction >= 0).all()


class TestStrategyRegistry:
    """Strategy Registry 테스트."""

    def test_list_strategies(self):
        """list_strategies() 테스트."""
        strategies = list_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) >= 2
        assert "tsmom" in strategies
        assert "adaptive-breakout" in strategies

    def test_get_nonexistent_strategy(self):
        """존재하지 않는 전략 조회 시 에러."""
        with pytest.raises(KeyError):
            get_strategy("nonexistent-strategy")
