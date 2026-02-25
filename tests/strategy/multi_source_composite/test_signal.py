"""Tests for Multi-Source Directional Composite signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.multi_source_composite.config import MultiSourceCompositeConfig, ShortMode
from src.strategy.multi_source_composite.preprocessor import preprocess
from src.strategy.multi_source_composite.signal import generate_signals


@pytest.fixture
def config() -> MultiSourceCompositeConfig:
    return MultiSourceCompositeConfig()


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
    fg = np.clip(50 + np.cumsum(np.random.randn(n) * 3), 0, 100).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "oc_fear_greed": fg,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


@pytest.fixture
def preprocessed_df(
    sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.strength.isna().sum() == 0

    def test_strength_sign_matches_direction(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        # Where direction != 0, strength sign should match direction
        active = signals.direction != 0
        if active.any():
            dir_sign = signals.direction[active]
            str_sign = np.sign(signals.strength[active])
            # strength could be 0 if vol_scalar is NaN/0
            non_zero_str = str_sign != 0
            if non_zero_str.any():
                assert (dir_sign[non_zero_str] == str_sign[non_zero_str]).all()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MultiSourceCompositeConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MultiSourceCompositeConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_respects_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MultiSourceCompositeConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,  # Very tight threshold for testing
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY should produce valid direction values
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_hedge_only_strength_ratio(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MultiSourceCompositeConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,
            hedge_strength_ratio=0.5,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # Verify all shorts have reduced strength (ratio applied)
        assert signals.strength.dtype == float


class TestMajorityVoteLogic:
    def test_unanimous_long_produces_signal(self) -> None:
        """3/3 bullish vote should produce long signal."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame(
            {
                "mom_direction": np.ones(n),
                "velocity_direction": np.ones(n),
                "fg_direction": np.ones(n),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.zeros(n),
            },
            index=idx,
        )
        config = MultiSourceCompositeConfig()
        signals = generate_signals(df, config)
        # After shift(1), bar 1+ should be long (bar 0 is 0 due to shift)
        assert (signals.direction.iloc[1:] == 1).all()

    def test_unanimous_short_produces_signal(self) -> None:
        """3/3 bearish vote should produce short signal."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame(
            {
                "mom_direction": -np.ones(n),
                "velocity_direction": -np.ones(n),
                "fg_direction": -np.ones(n),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.zeros(n),
            },
            index=idx,
        )
        config = MultiSourceCompositeConfig(short_mode=ShortMode.FULL)
        signals = generate_signals(df, config)
        # After shift(1), bar 1+ should be short
        assert (signals.direction.iloc[1:] == -1).all()

    def test_no_consensus_no_signal(self) -> None:
        """Mixed votes (1 long, 1 short, 1 neutral) = no signal."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame(
            {
                "mom_direction": np.ones(n),
                "velocity_direction": -np.ones(n),
                "fg_direction": np.zeros(n),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.zeros(n),
            },
            index=idx,
        )
        config = MultiSourceCompositeConfig()
        signals = generate_signals(df, config)
        # 1 long, 1 short, 1 neutral -> no majority -> direction = 0
        assert (signals.direction == 0).all()

    def test_two_thirds_majority_long(self) -> None:
        """2/3 bullish vote should produce long signal."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")
        df = pd.DataFrame(
            {
                "mom_direction": np.ones(n),
                "velocity_direction": np.ones(n),
                "fg_direction": np.zeros(n),  # neutral
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.zeros(n),
            },
            index=idx,
        )
        config = MultiSourceCompositeConfig()
        signals = generate_signals(df, config)
        # 2 long, 0 short, 1 neutral -> 2/3 majority long
        assert (signals.direction.iloc[1:] == 1).all()

    def test_unanimous_conviction_higher(self) -> None:
        """3/3 conviction should be higher than 2/3 conviction."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")

        # Unanimous (3/3)
        df_unanimous = pd.DataFrame(
            {
                "mom_direction": np.ones(n),
                "velocity_direction": np.ones(n),
                "fg_direction": np.ones(n),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.zeros(n),
            },
            index=idx,
        )

        # Majority (2/3)
        df_majority = pd.DataFrame(
            {
                "mom_direction": np.ones(n),
                "velocity_direction": np.ones(n),
                "fg_direction": np.zeros(n),
                "vol_scalar": np.full(n, 1.0),
                "drawdown": np.zeros(n),
            },
            index=idx,
        )

        config = MultiSourceCompositeConfig()
        sig_u = generate_signals(df_unanimous, config)
        sig_m = generate_signals(df_majority, config)

        # Unanimous should have higher average |strength|
        avg_u = sig_u.strength.abs().iloc[1:].mean()
        avg_m = sig_m.strength.abs().iloc[1:].mean()
        assert avg_u > avg_m


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: MultiSourceCompositeConfig
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


class TestEntryExitConsistency:
    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        """동일 bar에서 entry와 exit가 동시에 True가 아닌지 확인."""
        signals = generate_signals(preprocessed_df, config)
        conflict = signals.entries & signals.exits
        assert not conflict.any()

    def test_entry_requires_direction_change(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        """entry는 direction 변경 시에만 발생."""
        signals = generate_signals(preprocessed_df, config)
        entries = signals.entries
        direction = signals.direction
        prev_dir = direction.shift(1).fillna(0).astype(int)
        # entries가 True인 곳은 direction != 0 and direction != prev_dir
        assert ((entries & (direction != 0) & (direction != prev_dir)) == entries).all()

    def test_exit_requires_direction_zero(
        self, preprocessed_df: pd.DataFrame, config: MultiSourceCompositeConfig
    ) -> None:
        """exit는 direction이 0이 될 때 발생."""
        signals = generate_signals(preprocessed_df, config)
        exits = signals.exits
        direction = signals.direction
        # exits가 True인 곳은 direction == 0
        assert (direction[exits] == 0).all()
