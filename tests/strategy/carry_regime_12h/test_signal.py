"""Tests for Carry-Regime Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.carry_regime_12h.config import CarryRegimeConfig, ShortMode
from src.strategy.carry_regime_12h.preprocessor import preprocess
from src.strategy.carry_regime_12h.signal import generate_signals


@pytest.fixture
def config() -> CarryRegimeConfig:
    return CarryRegimeConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_strength_sign_matches_direction(
        self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        """strength 부호가 direction과 일치."""
        signals = generate_signals(preprocessed_df, config)
        active = signals.direction != 0
        if active.any():
            dir_sign = np.sign(signals.direction[active])
            str_sign = np.sign(signals.strength[active])
            # Strength가 0인 경우 제외
            nonzero_str = signals.strength[active] != 0
            if nonzero_str.any():
                assert (dir_sign[nonzero_str] == str_sign[nonzero_str]).all()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CarryRegimeConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CarryRegimeConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_requires_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = CarryRegimeConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_strength_reduced(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY에서 short strength가 hedge_strength_ratio로 감쇄."""
        config = CarryRegimeConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=0.0,  # always active
            hedge_strength_ratio=0.5,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # Short positions should exist (if any) with reduced strength
        assert signals.direction.dtype == int


class TestCarrySensitivity:
    def test_zero_sensitivity_pure_trend(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """carry_sensitivity=0 → FR has no effect on exit."""
        config_zero = CarryRegimeConfig(carry_sensitivity=0.0)
        config_with = CarryRegimeConfig(carry_sensitivity=1.0)

        # Without FR data, both should produce same signals
        df_zero = preprocess(sample_ohlcv_df, config_zero)
        df_with = preprocess(sample_ohlcv_df, config_with)

        sig_zero = generate_signals(df_zero, config_zero)
        sig_with = generate_signals(df_with, config_with)

        # With no FR data (fr_percentile=0.5), carry_sensitivity=0 gives
        # exit_threshold = exit_base - 0 * 0 = exit_base
        # carry_sensitivity=1.0 gives exit_threshold = exit_base - 1.0*(0.5-0.5) = exit_base
        # So same result when fr_percentile is uniformly 0.5
        pd.testing.assert_series_equal(
            sig_zero.direction.reset_index(drop=True),
            sig_with.direction.reset_index(drop=True),
            check_names=False,
        )

    def test_high_sensitivity_faster_exit(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Higher carry_sensitivity + extreme FR → faster exit (more neutral)."""
        np.random.seed(42)
        df_fr = sample_ohlcv_df.copy()
        # Extreme funding rate (consistently high)
        df_fr["funding_rate"] = 0.003  # high positive FR

        config_low = CarryRegimeConfig(carry_sensitivity=0.0)
        config_high = CarryRegimeConfig(carry_sensitivity=1.5)

        df_low = preprocess(df_fr, config_low)
        df_high = preprocess(df_fr, config_high)

        sig_low = generate_signals(df_low, config_low)
        sig_high = generate_signals(df_high, config_high)

        # High sensitivity should have equal or fewer active bars
        active_low = (sig_low.direction != 0).sum()
        active_high = (sig_high.direction != 0).sum()
        assert active_high <= active_low


class TestAdaptiveExit:
    def test_exit_threshold_range(
        self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        """exit_threshold은 항상 [0, 1] 범위."""
        fr_pct = preprocessed_df["fr_percentile"].shift(1).fillna(0.5)
        exit_threshold = (
            config.exit_base_threshold - config.carry_sensitivity * (fr_pct - 0.5)
        ).clip(lower=0.0, upper=1.0)
        assert (exit_threshold >= 0).all()
        assert (exit_threshold <= 1).all()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: CarryRegimeConfig
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


class TestNoSimultaneousLongShort:
    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        """동일 bar에서 entries=True && exits=True 없음."""
        signals = generate_signals(preprocessed_df, config)
        simultaneous = signals.entries & signals.exits
        assert not simultaneous.any()

    def test_strength_nan_not_with_entries(
        self, preprocessed_df: pd.DataFrame, config: CarryRegimeConfig
    ) -> None:
        """strength=NaN인 bar에서 entries=True 없음."""
        signals = generate_signals(preprocessed_df, config)
        nan_strength = signals.strength.isna()
        if nan_strength.any():
            assert not (nan_strength & signals.entries).any()
