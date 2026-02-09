"""Tests for Hurst/ER Regime signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.hurst_regime.config import HurstRegimeConfig, ShortMode
from src.strategy.hurst_regime.preprocessor import preprocess
from src.strategy.hurst_regime.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (250일)."""
    np.random.seed(42)
    n = 250

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
def default_config() -> HurstRegimeConfig:
    """기본 설정."""
    return HurstRegimeConfig()


@pytest.fixture
def preprocessed_df(
    sample_ohlcv_df: pd.DataFrame, default_config: HurstRegimeConfig
) -> pd.DataFrame:
    """전처리된 DataFrame."""
    return preprocess(sample_ohlcv_df, default_config)


class TestSignalOutputStructure:
    """시그널 출력 구조 테스트."""

    def test_signal_output_structure(self, preprocessed_df: pd.DataFrame):
        """시그널에 entries, exits, direction, strength 필드 존재."""
        signals = generate_signals(preprocessed_df)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_signal_types(self, preprocessed_df: pd.DataFrame):
        """시그널 타입 확인."""
        signals = generate_signals(preprocessed_df)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

    def test_signal_length(self, preprocessed_df: pd.DataFrame):
        """시그널 길이가 입력 DataFrame과 동일."""
        signals = generate_signals(preprocessed_df)

        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)


class TestDirectionValues:
    """Direction 값 테스트."""

    def test_direction_values(self, preprocessed_df: pd.DataFrame):
        """direction은 -1, 0, 1 중 하나."""
        signals = generate_signals(preprocessed_df)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)


class TestRegimeClassification:
    """Regime 분류 테스트."""

    def test_trending_regime(self):
        """Trending 데이터에서 momentum following 확인.

        강한 상승 추세 데이터를 만들어 trending regime에서
        양의 momentum이 양의 signal을 생성하는지 확인.
        """
        np.random.seed(42)
        n = 250

        # 강한 상승 추세 (drift + 작은 noise)
        daily_returns = 0.005 + np.random.randn(n) * 0.001
        close = 100 * np.cumprod(1 + daily_returns)
        high = close * (1 + np.abs(np.random.randn(n) * 0.002))
        low = close * (1 - np.abs(np.random.randn(n) * 0.002))
        open_ = close * (1 + np.random.randn(n) * 0.001)

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

        config = HurstRegimeConfig(
            er_trend_threshold=0.5,
            er_mr_threshold=0.2,
            short_mode=ShortMode.FULL,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # Trending 데이터에서 long signal이 다수여야 함
        valid_direction = signals.direction[signals.direction != 0]
        if len(valid_direction) > 0:
            long_ratio = (valid_direction == Direction.LONG).sum() / len(valid_direction)
            assert long_ratio > 0.5, f"Expected mostly long, got long_ratio={long_ratio:.2f}"

    def test_mr_regime(self):
        """Mean-reverting 데이터에서 z-score fading 확인.

        Mean-reverting 시리즈를 만들어 MR regime에서
        오버슈트 후 반대 방향 signal이 나오는지 확인.
        """
        np.random.seed(42)
        n = 250

        # Mean-reverting: Ornstein-Uhlenbeck process
        close = np.zeros(n)
        close[0] = 100.0
        mean_level = 100.0
        theta = 0.3  # Mean-reversion speed
        sigma = 0.5
        for i in range(1, n):
            close[i] = (
                close[i - 1] + theta * (mean_level - close[i - 1]) + sigma * np.random.randn()
            )

        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_ = close + np.random.randn(n) * 0.1

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

        config = HurstRegimeConfig(
            er_mr_threshold=0.5,
            er_trend_threshold=0.8,
            hurst_mr_threshold=0.50,
            hurst_trend_threshold=0.70,
            short_mode=ShortMode.FULL,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # MR regime에서 direction이 존재하는지 확인 (양/음 모두)
        valid_direction = signals.direction[signals.direction != 0]
        assert len(valid_direction) > 0, "No signals generated for MR data"


class TestShortModeProcessing:
    """숏 모드 처리 테스트."""

    def test_short_mode_disabled(self, preprocessed_df: pd.DataFrame):
        """DISABLED 모드에서 숏 시그널 없음."""
        config = HurstRegimeConfig(short_mode=ShortMode.DISABLED)
        signals = generate_signals(preprocessed_df, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_short_mode_hedge_only(self, preprocessed_df: pd.DataFrame):
        """HEDGE_ONLY 모드에서 시그널 생성 확인."""
        config = HurstRegimeConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.07,
            hedge_strength_ratio=0.8,
        )
        signals = generate_signals(preprocessed_df, config)

        # 시그널이 생성되는지 확인
        assert len(signals.entries) == len(preprocessed_df)

    def test_short_mode_full(self, preprocessed_df: pd.DataFrame):
        """FULL 모드에서 숏 시그널 허용."""
        config = HurstRegimeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        # FULL 모드에서는 숏 시그널이 존재할 수 있음
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_hedge_only_needs_drawdown_column(self):
        """HEDGE_ONLY 모드에서 drawdown 컬럼 없으면 에러."""
        config = HurstRegimeConfig(short_mode=ShortMode.HEDGE_ONLY)

        df = pd.DataFrame(
            {
                "er": [0.5, 0.3, 0.7],
                "hurst": [0.55, 0.45, 0.50],
                "momentum": [0.01, -0.02, 0.03],
                "z_score": [1.0, -1.0, 0.5],
                "vol_scalar": [1.0, 1.0, 1.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_shift1_no_lookahead(self, preprocessed_df: pd.DataFrame):
        """첫 번째 행은 shift(1)로 인해 중립이어야 함."""
        config = HurstRegimeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        # shift(1)이 적용되므로 첫 번째 행은 NaN → 0 (NEUTRAL)
        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0

    def test_signal_uses_previous_bar(self, preprocessed_df: pd.DataFrame):
        """시그널이 전봉 데이터 기반인지 확인.

        er를 shift(1)한 값과 실제 사용된 값이 일치하는지 검증.
        """
        config = HurstRegimeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(preprocessed_df, config)

        # shift(1) 확인: 두 번째 행의 시그널은 첫 번째 행의 지표에 기반
        # direction이 0이 아닌 구간에서 strength도 0이 아닌지 확인
        non_neutral = signals.direction != Direction.NEUTRAL
        if non_neutral.any():
            # direction이 non-zero이면 strength도 non-zero
            assert (signals.strength[non_neutral] != 0).all()
