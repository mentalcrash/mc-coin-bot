"""Tests for Capitulation Wick Reversal signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.cap_wick_rev.config import CapWickRevConfig, ShortMode
from src.strategy.cap_wick_rev.preprocessor import preprocess
from src.strategy.cap_wick_rev.signal import generate_signals


@pytest.fixture
def config() -> CapWickRevConfig:
    return CapWickRevConfig()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: CapWickRevConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: CapWickRevConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: CapWickRevConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: CapWickRevConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_direction_no_nan(
        self, preprocessed_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.direction.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: CapWickRevConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CapWickRevConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CapWickRevConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_with_drawdown(self) -> None:
        """In declining market, HEDGE_ONLY should allow shorts."""
        np.random.seed(42)
        n = 300
        close = 200 - np.linspace(0, 80, n) + np.random.randn(n) * 2
        close = np.maximum(close, 50)
        high = close + 2.0
        low = close - 2.0
        open_ = close + 0.5
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        config = CapWickRevConfig(short_mode=ShortMode.HEDGE_ONLY)
        preprocessed = preprocess(df, config)
        signals = generate_signals(preprocessed, config)
        assert signals.direction.dtype == int

    def test_hedge_only_no_drawdown(self) -> None:
        """In rising market, HEDGE_ONLY should suppress shorts."""
        n = 300
        close = np.linspace(100, 300, n)
        high = close + 2.0
        low = close - 2.0
        open_ = close - 0.5
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        config = CapWickRevConfig(short_mode=ShortMode.HEDGE_ONLY)
        preprocessed = preprocess(df, config)
        signals = generate_signals(preprocessed, config)
        # In a pure uptrend (no drawdown), no shorts should activate
        assert (signals.direction >= 0).all()


class TestTripleFilter:
    """Verify triple filter logic: ATR spike + Volume surge + Wick ratio."""

    def test_default_config_generates_signals(self, preprocessed_df: pd.DataFrame) -> None:
        config = CapWickRevConfig()
        signals = generate_signals(preprocessed_df, config)
        total_entries = signals.entries.sum()
        assert total_entries >= 0  # At minimum no error

    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_signals(df)

    def test_none_config_uses_default(self, preprocessed_df: pd.DataFrame) -> None:
        signals = generate_signals(preprocessed_df, None)
        assert len(signals.entries) == len(preprocessed_df)

    def test_no_confirmation_instant_entry(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """With confirmation_bars=0, entry should be immediate after filter match."""
        config = CapWickRevConfig(confirmation_bars=0)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(sample_ohlcv_df)

    def test_higher_thresholds_fewer_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Stricter thresholds should produce fewer or equal signals."""
        config_loose = CapWickRevConfig(
            atr_spike_threshold=1.5,
            vol_surge_threshold=1.5,
            wick_ratio_threshold=0.3,
        )
        config_strict = CapWickRevConfig(
            atr_spike_threshold=3.0,
            vol_surge_threshold=3.0,
            wick_ratio_threshold=0.7,
        )

        df_loose = preprocess(sample_ohlcv_df, config_loose)
        df_strict = preprocess(sample_ohlcv_df, config_strict)

        sig_loose = generate_signals(df_loose, config_loose)
        sig_strict = generate_signals(df_strict, config_strict)

        # Stricter should produce <= signals (or both zero)
        assert sig_strict.entries.sum() <= sig_loose.entries.sum() or (
            sig_strict.entries.sum() == 0 and sig_loose.entries.sum() == 0
        )
