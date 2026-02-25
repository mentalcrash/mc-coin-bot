"""Tests for Disposition CGO signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.disposition_cgo.config import DispositionCgoConfig, ShortMode
from src.strategy.disposition_cgo.preprocessor import preprocess
from src.strategy.disposition_cgo.signal import generate_signals


@pytest.fixture
def config() -> DispositionCgoConfig:
    return DispositionCgoConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: DispositionCgoConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_no_entry_with_nan_strength(
        self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig
    ) -> None:
        """strength=NaN인 bar에서 entries=True가 없어야 함."""
        signals = generate_signals(preprocessed_df, config)
        nan_mask = signals.strength.isna()
        assert not (signals.entries & nan_mask).any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DispositionCgoConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DispositionCgoConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_direction_range(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DispositionCgoConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_disabled_no_negative_strength(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DispositionCgoConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.strength >= 0).all()


class TestSignalLogic:
    def test_no_simultaneous_entry_and_exit(
        self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig
    ) -> None:
        """동일 bar에서 entry와 exit가 모두 True가 아닌지 확인."""
        signals = generate_signals(preprocessed_df, config)
        conflict = signals.entries & signals.exits
        assert not conflict.any()

    def test_entry_requires_direction_change(
        self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig
    ) -> None:
        """entry는 direction 변경 시에만 발생."""
        signals = generate_signals(preprocessed_df, config)
        prev_dir = signals.direction.shift(1).fillna(0).astype(int)
        entry_mask = signals.entries
        # entry가 True인 곳에서는 direction != prev_dir
        if entry_mask.any():
            assert (signals.direction[entry_mask] != prev_dir[entry_mask]).all()

    def test_strength_sign_matches_direction(
        self, preprocessed_df: pd.DataFrame, config: DispositionCgoConfig
    ) -> None:
        """strength 부호가 direction과 일치."""
        signals = generate_signals(preprocessed_df, config)
        nonzero = signals.strength != 0
        if nonzero.any():
            strength_sign = np.sign(signals.strength[nonzero])
            direction_sign = signals.direction[nonzero].astype(float)
            assert (strength_sign == direction_sign).all()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: DispositionCgoConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: DispositionCgoConfig
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
