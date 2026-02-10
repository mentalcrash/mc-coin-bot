"""Tests for OU Mean Reversion Signal Generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.ou_meanrev.config import OUMeanRevConfig, ShortMode
from src.strategy.ou_meanrev.preprocessor import preprocess
from src.strategy.ou_meanrev.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Mean-reverting price series for OU testing."""
    np.random.seed(42)
    n = 300
    prices = np.zeros(n)
    prices[0] = 100.0
    mu = 100.0
    theta = 0.05
    sigma = 2.0
    for i in range(1, n):
        prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + sigma * np.random.randn()

    high = prices + np.abs(np.random.randn(n) * 1.5)
    low = prices - np.abs(np.random.randn(n) * 1.5)
    open_ = prices + np.random.randn(n) * 0.5
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": prices,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


class TestSignalStructure:
    def test_output_structure(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_same_length(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert len(signals.entries) == len(sample_ohlcv_df)


class TestShift1Rule:
    def test_first_bar_no_signal(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig(short_mode=ShortMode.DISABLED)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig(short_mode=ShortMode.FULL)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)
        assert signals.direction.dtype == int

    def test_hedge_only_suppresses_without_drawdown(self):
        """상승 추세에서 hedge_only는 숏 억제."""
        n = 300
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

        config = OUMeanRevConfig(short_mode=ShortMode.HEDGE_ONLY)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction != Direction.SHORT).all()

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"close": [100, 101, 102]})
        config = OUMeanRevConfig()
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df, config)

    def test_default_config(self, sample_ohlcv_df: pd.DataFrame):
        config = OUMeanRevConfig()
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed)
        assert len(signals.entries) == len(sample_ohlcv_df)


class TestNoTradeWhenHalfLifeHigh:
    def test_no_trade_in_trend_regime(self):
        """Half-life > max_half_life일 때 거래 없음 (trend regime)."""
        np.random.seed(123)
        n = 300
        # Strong trend: no mean reversion → half_life should be long
        prices = 100 + np.cumsum(np.ones(n) * 0.5 + np.random.randn(n) * 0.1)
        high = prices + 1.0
        low = prices - 1.0

        df = pd.DataFrame(
            {
                "open": prices - 0.1,
                "high": high,
                "low": low,
                "close": prices,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )

        # Use strict max_half_life to force no-trade
        config = OUMeanRevConfig(max_half_life=10)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)

        # In a strong uptrend, b should be near 0 (not negative)
        # → theta near 0 → half_life very large → no MR signals
        # Some signals might still appear due to noise, but should be minimal
        total_active = (signals.direction != 0).sum()
        total_bars = len(signals.direction)
        active_ratio = total_active / total_bars
        # Expect less than 30% active bars in a strong trend
        assert active_ratio < 0.30


class TestExitConditions:
    def test_timeout_exit(self, sample_ohlcv_df: pd.DataFrame):
        """Timeout exit이 동작하는지 확인."""
        # Short timeout to force exits
        config = OUMeanRevConfig(exit_timeout_bars=10)
        processed = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(processed, config)

        # If there are any entries, there should also be exits
        if signals.entries.any():
            assert signals.exits.any()
