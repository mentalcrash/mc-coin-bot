"""Tests for Entropy Switch Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.entropy_switch.config import EntropySwitchConfig, ShortMode
from src.strategy.entropy_switch.preprocessor import preprocess
from src.strategy.entropy_switch.signal import generate_signals
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
        config = EntropySwitchConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_ohlcv_df)


class TestShift1Rule:
    def test_first_bar_no_signal(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert signals.direction.dtype == int

    def test_hedge_only_suppresses_without_drawdown(self) -> None:
        """상승 추세에서 HEDGE_ONLY는 숏을 억제."""
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

        config = EntropySwitchConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = EntropySwitchConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = EntropySwitchConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_no_adx_filter(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """ADX 필터 비활성화 시 adx 컬럼 없이도 동작."""
        config = EntropySwitchConfig(use_adx_filter=False, short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_adx_filter_reduces_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """ADX 필터 활성화 시 시그널이 줄어들거나 같아야 함."""
        config_no_adx = EntropySwitchConfig(
            use_adx_filter=False,
            short_mode=ShortMode.FULL,
        )
        config_with_adx = EntropySwitchConfig(
            use_adx_filter=True,
            short_mode=ShortMode.FULL,
        )

        processed_no_adx = preprocess(sample_ohlcv_df, config_no_adx)
        processed_with_adx = preprocess(sample_ohlcv_df, config_with_adx)

        signals_no_adx = generate_signals(processed_no_adx, config_no_adx)
        signals_with_adx = generate_signals(processed_with_adx, config_with_adx)

        # ADX 필터가 추가되면 시그널이 줄어들거나 같아야 함
        active_no_adx = (signals_no_adx.direction != 0).sum()
        active_with_adx = (signals_with_adx.direction != 0).sum()
        assert active_with_adx <= active_no_adx
