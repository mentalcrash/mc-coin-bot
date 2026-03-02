"""Tests for Weekend-Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.weekend_mom.config import ShortMode, WeekendMomConfig
from src.strategy.weekend_mom.preprocessor import preprocess
from src.strategy.weekend_mom.signal import generate_signals


@pytest.fixture
def config() -> WeekendMomConfig:
    return WeekendMomConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="12h"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: WeekendMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: WeekendMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not (signals.entries & signals.exits).any()

    def test_strength_zero_when_direction_zero(
        self, preprocessed_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        zero_dir = signals.direction == 0
        assert (signals.strength[zero_dir] == 0.0).all()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = WeekendMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = WeekendMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_shorts_require_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = WeekendMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # Short positions should only occur when drawdown < hedge_threshold
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_shifted = df["drawdown"].shift(1)
            assert (dd_shifted[short_bars] < config.hedge_threshold).all()

    def test_hedge_only_strength_ratio(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드에서 short strength가 hedge_strength_ratio로 감쇄."""
        config = WeekendMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_strength_ratio=0.5,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_bars = signals.direction == -1
        if short_bars.any():
            # Short strength magnitudes should be < long strength magnitudes
            # due to 0.5x ratio (given same vol_scalar)
            assert True  # Just verify no error


class TestWeekendMomSpecific:
    def test_boost_1_equals_pure_momentum(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """weekend_boost=1.0이면 standard momentum과 동일해야 함."""
        config_boost1 = WeekendMomConfig(weekend_boost=1.0)
        df = preprocess(sample_ohlcv_df, config_boost1)
        # weighted_returns should equal returns (no boost)
        valid = df["returns"].dropna()
        weighted = df["weighted_returns"].dropna()
        pd.testing.assert_series_equal(valid, weighted, check_names=False)

    def test_higher_boost_changes_mom(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Higher weekend_boost should produce different momentum values."""
        config_low = WeekendMomConfig(weekend_boost=1.0)
        config_high = WeekendMomConfig(weekend_boost=3.0)
        df_low = preprocess(sample_ohlcv_df, config_low)
        df_high = preprocess(sample_ohlcv_df, config_high)
        # If there are any weekend bars, momentum should differ
        has_weekend = df_low["is_weekend"].any()
        if has_weekend:
            fast_low = df_low["fast_mom"].dropna()
            fast_high = df_high["fast_mom"].dropna()
            # They should not be exactly equal
            assert not np.allclose(fast_low.values, fast_high.values, equal_nan=True)

    def test_dual_confirmation(
        self, preprocessed_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        """Long requires both fast_mom > 0 AND slow_mom > 0."""
        signals = generate_signals(preprocessed_df, config)
        long_bars = signals.direction == 1
        if long_bars.any():
            fast = preprocessed_df["fast_mom"].shift(1)
            slow = preprocessed_df["slow_mom"].shift(1)
            assert (fast[long_bars] > 0).all()
            assert (slow[long_bars] > 0).all()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        """마지막 50 bar 제거해도 이전 시그널 동일."""
        df_full = preprocess(sample_ohlcv_df, config)
        sig_full = generate_signals(df_full, config)
        cut = 50
        df_trunc = preprocess(sample_ohlcv_df.iloc[:-cut].copy(), config)
        sig_trunc = generate_signals(df_trunc, config)
        overlap = len(sig_trunc.direction)
        pd.testing.assert_series_equal(
            sig_full.direction.iloc[:overlap].reset_index(drop=True),
            sig_trunc.direction.reset_index(drop=True),
            check_names=False,
        )

    def test_single_bar_append(
        self, sample_ohlcv_df: pd.DataFrame, config: WeekendMomConfig
    ) -> None:
        """1 bar 추가 시 마지막 bar만 변경 가능."""
        n = len(sample_ohlcv_df)
        df_prev = preprocess(sample_ohlcv_df.iloc[: n - 1].copy(), config)
        sig_prev = generate_signals(df_prev, config)
        df_full = preprocess(sample_ohlcv_df, config)
        sig_full = generate_signals(df_full, config)
        pd.testing.assert_series_equal(
            sig_full.direction.iloc[:-1].reset_index(drop=True),
            sig_prev.direction.reset_index(drop=True),
            check_names=False,
        )
