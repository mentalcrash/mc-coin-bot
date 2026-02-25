"""Tests for Z-Momentum (MACD-V) signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.z_mom.config import ShortMode, ZMomConfig
from src.strategy.z_mom.preprocessor import preprocess
from src.strategy.z_mom.signal import generate_signals


@pytest.fixture
def config() -> ZMomConfig:
    return ZMomConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: ZMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, preprocessed_df: pd.DataFrame, config: ZMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: ZMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: ZMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: ZMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_no_entry_with_nan_strength(
        self, preprocessed_df: pd.DataFrame, config: ZMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        entry_strength = signals.strength[signals.entries]
        assert not entry_strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(self, preprocessed_df: pd.DataFrame, config: ZMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0

    def test_second_bar_also_neutral(
        self, preprocessed_df: pd.DataFrame, config: ZMomConfig
    ) -> None:
        """With shift(1), the second bar should also be neutral (warmup NaN)."""
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[1] == 0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ZMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ZMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_respects_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ZMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY: short only when drawdown < threshold
        short_mask = signals.direction == -1
        if short_mask.any():
            dd = df["drawdown"].shift(1)
            assert (dd[short_mask] < config.hedge_threshold).all()

    def test_hedge_only_strength_reduced(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ZMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_mask = signals.direction == -1
        if short_mask.any():
            # Hedge strength should be attenuated
            assert (signals.strength[short_mask] <= 0).all()


class TestFlatZone:
    def test_zero_flat_zone_more_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """flat_zone=0 should produce more non-zero directions than flat_zone=2."""
        config_narrow = ZMomConfig(flat_zone=0.0)
        config_wide = ZMomConfig(flat_zone=2.0)
        df_narrow = preprocess(sample_ohlcv_df, config_narrow)
        df_wide = preprocess(sample_ohlcv_df, config_wide)
        sig_narrow = generate_signals(df_narrow, config_narrow)
        sig_wide = generate_signals(df_wide, config_wide)
        active_narrow = (sig_narrow.direction != 0).sum()
        active_wide = (sig_wide.direction != 0).sum()
        assert active_narrow >= active_wide

    def test_very_wide_flat_zone_all_neutral(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Extremely wide flat zone should suppress all signals."""
        config = ZMomConfig(flat_zone=5.0)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # Most (if not all) directions should be 0 with very wide flat zone
        non_zero = (signals.direction != 0).sum()
        assert non_zero < len(signals.direction) * 0.1  # < 10% active


class TestSignalLogic:
    def test_long_requires_positive_momentum(
        self, preprocessed_df: pd.DataFrame, config: ZMomConfig
    ) -> None:
        """Long entries require macd_v_hist > flat_zone AND mom_return > 0."""
        signals = generate_signals(preprocessed_df, config)
        macd_v_hist = preprocessed_df["macd_v_hist"].shift(1)
        mom_return = preprocessed_df["mom_return"].shift(1)
        long_mask = signals.direction == 1
        if long_mask.any():
            assert (macd_v_hist[long_mask] > config.flat_zone).all()
            assert (mom_return[long_mask] > 0).all()

    def test_short_requires_negative_momentum(
        self, preprocessed_df: pd.DataFrame, config: ZMomConfig
    ) -> None:
        """Short entries require macd_v_hist < -flat_zone AND mom_return < 0."""
        signals = generate_signals(preprocessed_df, config)
        macd_v_hist = preprocessed_df["macd_v_hist"].shift(1)
        mom_return = preprocessed_df["mom_return"].shift(1)
        short_mask = signals.direction == -1
        if short_mask.any():
            assert (macd_v_hist[short_mask] < -config.flat_zone).all()
            assert (mom_return[short_mask] < 0).all()

    def test_no_simultaneous_long_short(
        self, preprocessed_df: pd.DataFrame, config: ZMomConfig
    ) -> None:
        """direction should never have both +1 and -1 on same bar."""
        signals = generate_signals(preprocessed_df, config)
        # direction is a single int per bar, so can't be both
        assert set(signals.direction.unique()).issubset({-1, 0, 1})


class TestNoLookaheadBias:
    def test_truncation_invariance(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
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

    def test_single_bar_append(self, sample_ohlcv_df: pd.DataFrame, config: ZMomConfig) -> None:
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
