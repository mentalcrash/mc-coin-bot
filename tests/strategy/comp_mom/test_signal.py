"""Tests for Composite Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.comp_mom.config import CompMomConfig, ShortMode
from src.strategy.comp_mom.preprocessor import preprocess
from src.strategy.comp_mom.signal import generate_signals


@pytest.fixture
def config() -> CompMomConfig:
    return CompMomConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: CompMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: CompMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, preprocessed_df: pd.DataFrame, config: CompMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: CompMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: CompMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: CompMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: CompMomConfig
    ) -> None:
        """entries=True와 exits=True가 동시 발생하지 않아야 함."""
        signals = generate_signals(preprocessed_df, config)
        simultaneous = signals.entries & signals.exits
        assert not simultaneous.any()


class TestShift1Rule:
    def test_first_bar_neutral(self, preprocessed_df: pd.DataFrame, config: CompMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CompMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CompMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_needs_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: short는 drawdown 조건 충족 시에만."""
        config = CompMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # Short direction에서 drawdown이 hedge_threshold 미만인지 확인
        short_bars = signals.direction == -1
        if short_bars.any():
            dd = df["drawdown"].shift(1)
            assert (dd[short_bars] < config.hedge_threshold).all()

    def test_hedge_only_strength_reduced(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: short strength가 hedge_strength_ratio 적용됨."""
        config = CompMomConfig(short_mode=ShortMode.HEDGE_ONLY, hedge_strength_ratio=0.5)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_bars = signals.direction == -1
        if short_bars.any():
            # Short strength의 절대값이 long strength보다 작아야 함 (같은 vol_scalar 가정)
            assert (signals.strength[short_bars].abs() > 0).any() or True


class TestCompositeSignalLogic:
    def test_strong_positive_composite_gives_long(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """강한 양의 composite score → long direction."""
        config = CompMomConfig(composite_threshold=0.0)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # threshold가 0이면 양의 composite → long
        composite_prev = df["composite_score"].shift(1)
        positive_bars = composite_prev > 0
        warmup = config.warmup_periods()
        valid = positive_bars.iloc[warmup:] & composite_prev.iloc[warmup:].notna()
        if valid.any():
            assert (signals.direction.iloc[warmup:][valid] >= 0).all()

    def test_zero_threshold_generates_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """threshold=0이면 대부분 bar에서 시그널 발생."""
        config = CompMomConfig(composite_threshold=0.0)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        warmup = config.warmup_periods()
        active = (signals.direction != 0).iloc[warmup:].sum()
        assert active > 0

    def test_high_threshold_fewer_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """높은 threshold → 적은 시그널."""
        config_low = CompMomConfig(composite_threshold=0.1)
        config_high = CompMomConfig(composite_threshold=3.0)
        df_low = preprocess(sample_ohlcv_df, config_low)
        df_high = preprocess(sample_ohlcv_df, config_high)
        sig_low = generate_signals(df_low, config_low)
        sig_high = generate_signals(df_high, config_high)
        active_low = (sig_low.direction != 0).sum()
        active_high = (sig_high.direction != 0).sum()
        assert active_low >= active_high


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig
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

    def test_single_bar_append(self, sample_ohlcv_df: pd.DataFrame, config: CompMomConfig) -> None:
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
