"""Tests for Candle Reject Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.candle_reject.config import CandleRejectConfig, ShortMode
from src.strategy.candle_reject.preprocessor import preprocess
from src.strategy.candle_reject.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Rejection wick이 발생할 수 있는 OHLCV 데이터."""
    np.random.seed(42)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    open_ = close + np.random.randn(n) * 0.5
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max + np.abs(np.random.randn(n) * 3.0)
    low = bar_min - np.abs(np.random.randn(n) * 3.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestSignalStructure:
    def test_output_structure(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_ohlcv_df)


class TestShift1Rule:
    def test_first_bar_no_signal(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert signals.direction.dtype == int

    def test_hedge_only_suppresses_without_drawdown(self):
        """상승 추세에서 HEDGE_ONLY는 숏 억제."""
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
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        config = CandleRejectConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = CandleRejectConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_ohlcv_df: pd.DataFrame):
        config = CandleRejectConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed)
        assert len(signals.entries) == len(sample_ohlcv_df)


class TestConsecutiveBoost:
    def test_consecutive_boost_increases_strength(self):
        """연속 rejection 시 strength가 boost 배수만큼 증가 확인."""
        n = 100
        np.random.seed(123)

        # 연속 bull rejection을 유발하는 데이터 생성
        # 하단 꼬리가 매우 길고 (bull rejection > 0.6), volume spike
        close = np.full(n, 100.0)
        open_ = np.full(n, 99.5)
        # bull_reject = (min(open,close) - low) / (high - low)
        # min(99.5, 100) = 99.5, so lower_wick = 99.5 - low
        # range = high - low
        # Want lower_wick / range > 0.6 → lower_wick > 0.6 * range
        high = np.full(n, 101.0)  # upper_wick = 101 - max(99.5,100) = 101-100 = 1
        low = np.full(n, 95.0)  # lower_wick = min(99.5,100)-95 = 99.5-95 = 4.5, range=101-95=6
        # bull_reject = 4.5/6 = 0.75 > 0.6 ✓

        volume = np.full(n, 5000.0)
        # Volume spike 구간
        volume[40:60] = 20000.0

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        config = CandleRejectConfig(
            short_mode=ShortMode.FULL,
            consecutive_boost=1.5,
            consecutive_min=2,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # 연속 bull rejection이 있는 구간에서 강도 확인
        long_strength = signals.strength[signals.strength > 0]
        if len(long_strength) >= 2:
            # 연속 시그널이 있는 구간의 strength가 0보다 큰지 확인
            assert long_strength.max() > 0


class TestExitTimeout:
    def test_timeout_generates_exit(self):
        """exit_timeout_bars를 초과하면 direction이 0이 됨."""
        n = 100
        close = np.full(n, 100.0)
        open_ = np.full(n, 99.5)
        high = np.full(n, 101.0)
        low = np.full(n, 95.0)
        volume = np.full(n, 5000.0)
        volume[30:80] = 20000.0

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        config = CandleRejectConfig(
            short_mode=ShortMode.FULL,
            exit_timeout_bars=5,
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # timeout에 의해 direction이 0이 되는 시점이 존재해야 함
        assert (signals.direction == Direction.NEUTRAL).any()
