"""Tests for Kalman Trend Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.kalman_trend.config import KalmanTrendConfig, ShortMode
from src.strategy.kalman_trend.preprocessor import preprocess
from src.strategy.kalman_trend.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestSignalStructure:
    def test_output_structure(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = KalmanTrendConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = KalmanTrendConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = KalmanTrendConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = KalmanTrendConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_ohlcv_df)


class TestShift1Rule:
    def test_first_bar_no_signal(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """첫 번째 bar에서는 shift(1)로 인해 시그널이 없어야 함."""
        config = KalmanTrendConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """DISABLED 모드에서 숏 시그널 없어야 함."""
        config = KalmanTrendConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """FULL 모드에서 direction dtype이 int여야 함."""
        config = KalmanTrendConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert signals.direction.dtype == int

    def test_hedge_only_suppresses_without_drawdown(self) -> None:
        """드로다운 없는 상승 추세에서 HEDGE_ONLY 숏 억제 확인."""
        n = 200
        close = np.linspace(100, 200, n)
        high = close + 2.0
        low = close - 2.0

        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = KalmanTrendConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_missing_columns_raises(self) -> None:
        """필수 컬럼 누락 시 에러."""
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = KalmanTrendConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """기본 config로 시그널 생성."""
        config = KalmanTrendConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed)
        assert len(signals.entries) == len(sample_ohlcv_df)


class TestEntryExitLogic:
    def test_entry_on_direction_change(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """direction 변경 시 entry 시그널 확인."""
        config = KalmanTrendConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        direction = signals.direction
        prev_direction = direction.shift(1).fillna(0)

        # entry: direction이 LONG/SHORT로 새로 진입할 때
        expected_long_entry = (direction == Direction.LONG) & (prev_direction != Direction.LONG)
        expected_short_entry = (direction == Direction.SHORT) & (prev_direction != Direction.SHORT)
        expected_entries = expected_long_entry | expected_short_entry

        pd.testing.assert_series_equal(
            signals.entries,
            expected_entries,
            check_names=False,
        )

    def test_exit_on_neutral_or_reversal(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """중립 전환 또는 반전 시 exit 시그널 확인."""
        config = KalmanTrendConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        direction = signals.direction
        prev_direction = direction.shift(1).fillna(0)

        to_neutral = (direction == Direction.NEUTRAL) & (prev_direction != Direction.NEUTRAL)
        reversal = direction * prev_direction < 0

        expected_exits = to_neutral | reversal

        pd.testing.assert_series_equal(
            signals.exits,
            expected_exits,
            check_names=False,
        )
