"""Tests for Trend Quality Momentum (TQ-Mom) signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.tq_mom.config import ShortMode, TqMomConfig
from src.strategy.tq_mom.preprocessor import preprocess
from src.strategy.tq_mom.signal import generate_signals


@pytest.fixture
def config() -> TqMomConfig:
    return TqMomConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: TqMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: TqMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, preprocessed_df: pd.DataFrame, config: TqMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: TqMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: TqMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_strength(self, preprocessed_df: pd.DataFrame, config: TqMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(self, preprocessed_df: pd.DataFrame, config: TqMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TqMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TqMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_respects_threshold(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TqMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_at_shorts = df["drawdown"].shift(1)[short_bars]
            assert (dd_at_shorts < config.hedge_threshold).all()


class TestQualityGate:
    """Hurst + FD quality gate 검증."""

    def test_high_hurst_low_fd_entry(self, preprocessed_df: pd.DataFrame) -> None:
        """High Hurst + Low FD → quality pass → entry possible."""
        config = TqMomConfig(hurst_threshold=0.55, fd_threshold=1.4)
        df = preprocessed_df.copy()
        df["hurst"] = 0.7  # strong trend persistence
        df["fd"] = 1.1  # orderly movement
        df["price_mom"] = 0.05  # positive momentum
        signals = generate_signals(df, config)
        # After shift(1), entries should be possible
        active = signals.direction != 0
        if active.any():
            assert signals.strength[active].abs().mean() > 0

    def test_low_hurst_no_entry(self, preprocessed_df: pd.DataFrame) -> None:
        """Low Hurst → no quality pass → no entry."""
        config = TqMomConfig(hurst_threshold=0.55)
        df = preprocessed_df.copy()
        df["hurst"] = 0.45  # no trend persistence
        df["fd"] = 1.2  # orderly
        df["price_mom"] = 0.05
        signals = generate_signals(df, config)
        # After shift(1), all should be neutral (hurst below threshold)
        assert (signals.direction.iloc[2:] == 0).all()

    def test_high_fd_no_entry(self, preprocessed_df: pd.DataFrame) -> None:
        """High FD → no quality pass → no entry."""
        config = TqMomConfig(fd_threshold=1.4)
        df = preprocessed_df.copy()
        df["hurst"] = 0.7
        df["fd"] = 1.6  # too complex / random
        df["price_mom"] = 0.05
        signals = generate_signals(df, config)
        assert (signals.direction.iloc[2:] == 0).all()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: TqMomConfig
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

    def test_single_bar_append(self, sample_ohlcv_df: pd.DataFrame, config: TqMomConfig) -> None:
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
