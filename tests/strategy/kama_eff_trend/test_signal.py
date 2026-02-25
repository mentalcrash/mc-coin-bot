"""Tests for KAMA Efficiency Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.kama_eff_trend.config import KamaEffTrendConfig, ShortMode
from src.strategy.kama_eff_trend.preprocessor import preprocess
from src.strategy.kama_eff_trend.signal import generate_signals


@pytest.fixture
def config() -> KamaEffTrendConfig:
    return KamaEffTrendConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: KamaEffTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: KamaEffTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_strength_sign_matches_direction(
        self, preprocessed_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        mask = signals.direction != 0
        if mask.any():
            direction_sign = np.sign(signals.direction[mask])
            strength_sign = np.sign(signals.strength[mask])
            # Where strength is non-zero, sign should match direction
            nonzero_strength = signals.strength[mask] != 0
            if nonzero_strength.any():
                assert (direction_sign[nonzero_strength] == strength_sign[nonzero_strength]).all()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = KamaEffTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = KamaEffTrendConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_shorts_require_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = KamaEffTrendConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # Shorts only when drawdown < hedge_threshold
        short_mask = signals.direction == -1
        if short_mask.any():
            dd_at_short = df["drawdown"].shift(1)[short_mask]
            assert (dd_at_short < config.hedge_threshold).all()

    def test_hedge_only_strength_reduced(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config_full = KamaEffTrendConfig(short_mode=ShortMode.FULL)
        config_hedge = KamaEffTrendConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.001,  # Very permissive to allow shorts
        )
        df_full = preprocess(sample_ohlcv_df, config_full)
        sig_full = generate_signals(df_full, config_full)
        df_hedge = preprocess(sample_ohlcv_df, config_hedge)
        sig_hedge = generate_signals(df_hedge, config_hedge)

        # Hedge mode should have equal or less short strength magnitude
        short_mask_hedge = sig_hedge.direction == -1
        if short_mask_hedge.any():
            avg_hedge = sig_hedge.strength[short_mask_hedge].abs().mean()
            short_mask_full = sig_full.direction == -1
            if short_mask_full.any():
                avg_full = sig_full.strength[short_mask_full].abs().mean()
                assert avg_hedge <= avg_full + 1e-10


class TestErThresholdFilter:
    def test_er_below_threshold_no_signal(
        self, preprocessed_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        """ER < threshold인 bar에서는 direction = 0."""
        signals = generate_signals(preprocessed_df, config)
        er_shifted = preprocessed_df["er"].shift(1)
        low_er_mask = er_shifted < config.er_threshold
        # Remove first bar (NaN from shift)
        valid_mask = low_er_mask & er_shifted.notna()
        if valid_mask.any():
            assert (signals.direction[valid_mask] == 0).all()


class TestConviction:
    def test_conviction_bounded(
        self, preprocessed_df: pd.DataFrame, config: KamaEffTrendConfig
    ) -> None:
        """Strength magnitude bounded by vol_scalar * 1.0 (max conviction)."""
        signals = generate_signals(preprocessed_df, config)
        vol_scalar = preprocessed_df["vol_scalar"].shift(1).fillna(0)
        active = signals.direction != 0
        if active.any():
            max_possible = vol_scalar[active].abs()
            actual = signals.strength[active].abs()
            assert (actual <= max_possible + 1e-10).all()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: KamaEffTrendConfig
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
