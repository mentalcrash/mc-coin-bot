"""Tests for Vol-Adaptive Trend signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.types import Direction
from src.strategy.vol_adaptive.config import ShortMode, VolAdaptiveConfig
from src.strategy.vol_adaptive.preprocessor import preprocess
from src.strategy.vol_adaptive.signal import generate_signals


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


@pytest.fixture
def default_config() -> VolAdaptiveConfig:
    """기본 VolAdaptiveConfig."""
    return VolAdaptiveConfig()


class TestSignalOutputStructure:
    """시그널 출력 구조 테스트."""

    def test_signal_output_structure(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: VolAdaptiveConfig,
    ) -> None:
        """시그널에 entries, exits, direction, strength 필드 존재."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

        assert len(signals.entries) == len(sample_ohlcv_df)
        assert len(signals.exits) == len(sample_ohlcv_df)
        assert len(signals.direction) == len(sample_ohlcv_df)
        assert len(signals.strength) == len(sample_ohlcv_df)

    def test_direction_values(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: VolAdaptiveConfig,
    ) -> None:
        """direction은 -1, 0, 1 중 하나의 부분집합."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )


class TestADXFilter:
    """ADX 필터 테스트."""

    def test_adx_below_threshold_flat(self) -> None:
        """ADX가 threshold 미만이면 direction은 0이어야 함."""
        n = 200
        np.random.seed(123)

        # 랜덤워크 (약한 추세) → ADX 낮은 데이터
        close = 100 + np.cumsum(np.random.randn(n) * 0.1)
        high = close + 0.05
        low = close - 0.05
        open_ = close + np.random.randn(n) * 0.01

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        # 매우 높은 ADX threshold로 모든 시그널 필터링
        config = VolAdaptiveConfig(adx_threshold=40.0)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # warmup 이후 ADX가 threshold 미만인 구간에서 direction = 0
        warmup = config.warmup_periods()
        adx_values = processed["adx"].iloc[warmup:]
        direction_after_warmup = signals.direction.iloc[warmup:]

        # ADX < threshold인 곳에서 direction이 0이어야 함
        low_adx_mask = adx_values.shift(1) <= config.adx_threshold
        if low_adx_mask.any():
            low_adx_directions = direction_after_warmup[low_adx_mask]
            assert (low_adx_directions == 0).all()


class TestEMACrossover:
    """EMA crossover 방향 테스트."""

    def test_ema_crossover_direction(self) -> None:
        """강한 상승 추세에서 롱 시그널 기대."""
        n = 200
        np.random.seed(99)

        # 강한 상승 추세 + 약간의 노이즈 (ADX가 의미있는 값을 가지도록)
        trend = np.linspace(0, 150, n)
        noise = np.cumsum(np.random.randn(n) * 1.5)
        close = 100 + trend + noise
        high = close + np.abs(np.random.randn(n) * 3) + 1.0
        low = close - np.abs(np.random.randn(n) * 3) - 1.0
        open_ = close - 0.5

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(3000, 8000, n).astype(float),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = VolAdaptiveConfig(
            ema_fast=5,
            ema_slow=20,
            adx_threshold=10.0,  # 낮은 threshold로 필터링 완화
            rsi_upper=40.0,  # 낮은 RSI threshold로 확인 완화
            rsi_period=10,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # warmup 이후 롱 시그널이 존재해야 함
        warmup = config.warmup_periods()
        direction_after = signals.direction.iloc[warmup:]

        # 강한 상승 추세이므로 1 (롱) 시그널이 존재
        assert (direction_after == Direction.LONG).any()


class TestShortModes:
    """숏 모드 테스트."""

    def test_short_mode_disabled(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """DISABLED 모드에서 숏 시그널 없음."""
        config = VolAdaptiveConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_short_mode_hedge_only(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ) -> None:
        """HEDGE_ONLY 모드에서 드로다운 임계값에 따라 숏 제어."""
        config = VolAdaptiveConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.07,
            hedge_strength_ratio=0.8,
        )
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        # 시그널이 정상 생성됨
        assert len(signals.entries) == len(sample_ohlcv_df)

        # 숏 시그널이 있다면, drawdown < threshold인 구간에서만 존재해야 함
        short_mask = signals.direction == Direction.SHORT
        if short_mask.any():
            drawdown_at_short = processed["drawdown"][short_mask]
            assert (drawdown_at_short < config.hedge_threshold).all()


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_shift1_no_lookahead(
        self,
        sample_ohlcv_df: pd.DataFrame,
        default_config: VolAdaptiveConfig,
    ) -> None:
        """첫 번째 행은 shift(1)로 인해 중립이어야 함."""
        processed = preprocess(sample_ohlcv_df, default_config)
        signals = generate_signals(processed, default_config)

        # 첫 행은 shift(1) 결과 NaN → 0으로 채워짐
        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0
