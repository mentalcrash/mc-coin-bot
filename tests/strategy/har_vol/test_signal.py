"""Tests for HAR Volatility Overlay signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.har_vol.config import HARVolConfig
from src.strategy.har_vol.preprocessor import preprocess
from src.strategy.har_vol.signal import generate_signals
from src.strategy.tsmom.config import ShortMode
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame 생성 (400일)."""
    np.random.seed(42)
    n = 400

    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5) + 0.01
    low = close - np.abs(np.random.randn(n) * 1.5) - 0.01
    open_ = close + np.random.randn(n) * 0.5

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="D"),
    )

    return df


@pytest.fixture
def small_config() -> HARVolConfig:
    """작은 윈도우의 HAR Vol Config (빠른 테스트용)."""
    return HARVolConfig(
        daily_window=1,
        weekly_window=3,
        monthly_window=15,
        training_window=60,
    )


class TestSignalOutputStructure:
    """시그널 출력 구조 테스트."""

    def test_output_structure(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HARVolConfig,
    ):
        """시그널에 entries, exits, direction, strength 필드 존재."""
        processed = preprocess(sample_ohlcv_df, small_config)
        signals = generate_signals(processed, small_config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

        assert len(signals.entries) == len(processed)
        assert len(signals.exits) == len(processed)
        assert len(signals.direction) == len(processed)
        assert len(signals.strength) == len(processed)

    def test_direction_values(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HARVolConfig,
    ):
        """direction은 -1, 0, 1 중 하나."""
        processed = preprocess(sample_ohlcv_df, small_config)
        signals = generate_signals(processed, small_config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_entries_exits_bool(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HARVolConfig,
    ):
        """entries와 exits는 bool dtype."""
        processed = preprocess(sample_ohlcv_df, small_config)
        signals = generate_signals(processed, small_config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool


class TestShift1Rule:
    """Shift(1) Rule (미래 참조 편향 방지) 테스트."""

    def test_shift1_first_row_neutral(
        self,
        sample_ohlcv_df: pd.DataFrame,
        small_config: HARVolConfig,
    ):
        """첫 번째 행은 shift(1)로 인해 중립이어야 함."""
        processed = preprocess(sample_ohlcv_df, small_config)
        signals = generate_signals(processed, small_config)

        assert signals.direction.iloc[0] == Direction.NEUTRAL
        assert signals.strength.iloc[0] == 0.0


class TestVolSurpriseSignals:
    """Vol surprise 기반 시그널 테스트."""

    def test_vol_surprise_positive_momentum(self):
        """양의 vol surprise → momentum (returns 방향 추종)."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        config = HARVolConfig(
            vol_surprise_threshold=0.0,
            short_mode=ShortMode.FULL,
            training_window=60,
            monthly_window=15,
        )

        # 합성 데이터: positive vol_surprise + positive returns
        df = pd.DataFrame(
            {
                "vol_surprise": np.full(n, 0.05),  # positive surprise
                "returns": np.full(n, 0.01),  # positive returns
                "vol_scalar": np.full(n, 1.0),
            },
            index=idx,
        )

        signals = generate_signals(df, config)

        # shift(1) 이후 유효한 구간에서 long direction
        valid_direction = signals.direction.iloc[2:]
        long_count = (valid_direction == Direction.LONG).sum()
        assert long_count > 0, "Positive vol surprise + positive returns should produce LONG"

    def test_vol_surprise_negative_mean_reversion(self):
        """음의 vol surprise → mean-reversion (returns 반대 방향)."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        config = HARVolConfig(
            vol_surprise_threshold=0.0,
            short_mode=ShortMode.FULL,
            training_window=60,
            monthly_window=15,
        )

        # 합성 데이터: negative vol_surprise + positive returns → SHORT (mean-reversion)
        df = pd.DataFrame(
            {
                "vol_surprise": np.full(n, -0.05),  # negative surprise
                "returns": np.full(n, 0.01),  # positive returns → MR says SHORT
                "vol_scalar": np.full(n, 1.0),
            },
            index=idx,
        )

        signals = generate_signals(df, config)

        # shift(1) 이후 유효한 구간에서 short direction (mean-reversion)
        valid_direction = signals.direction.iloc[2:]
        short_count = (valid_direction == Direction.SHORT).sum()
        assert short_count > 0, (
            "Negative vol surprise + positive returns should produce SHORT (mean-reversion)"
        )

    def test_threshold_filter(self):
        """threshold 내의 vol surprise → neutral."""
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        config = HARVolConfig(
            vol_surprise_threshold=0.1,  # 높은 threshold
            short_mode=ShortMode.FULL,
            training_window=60,
            monthly_window=15,
        )

        # 합성 데이터: vol_surprise가 threshold 이내
        df = pd.DataFrame(
            {
                "vol_surprise": np.full(n, 0.05),  # < threshold (0.1)
                "returns": np.full(n, 0.01),
                "vol_scalar": np.full(n, 1.0),
            },
            index=idx,
        )

        signals = generate_signals(df, config)

        # |vol_surprise| < threshold → 모두 neutral
        valid_direction = signals.direction.iloc[2:]
        neutral_count = (valid_direction == Direction.NEUTRAL).sum()
        assert neutral_count == len(valid_direction), (
            "Vol surprise within threshold should be neutral"
        )


class TestShortModeSignals:
    """숏 모드별 시그널 테스트."""

    def test_full_short_mode(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """FULL 모드에서 숏 시그널 허용."""
        config = HARVolConfig(
            short_mode=ShortMode.FULL,
            daily_window=1,
            weekly_window=3,
            monthly_window=15,
            training_window=60,
        )
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_disabled_mode(
        self,
        sample_ohlcv_df: pd.DataFrame,
    ):
        """DISABLED 모드에서 숏 시그널 없음."""
        config = HARVolConfig(
            short_mode=ShortMode.DISABLED,
            daily_window=1,
            weekly_window=3,
            monthly_window=15,
            training_window=60,
        )
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()


class TestMissingColumns:
    """필수 컬럼 누락 테스트."""

    def test_missing_columns_raises(self):
        """필수 컬럼 누락 시 ValueError."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)

    def test_missing_vol_surprise_raises(self):
        """vol_surprise 누락 시 ValueError."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "returns": np.zeros(n),
                "vol_scalar": np.ones(n),
            },
            index=idx,
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)
