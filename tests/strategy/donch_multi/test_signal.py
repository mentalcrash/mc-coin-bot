"""Tests for Donchian Multi-Scale signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.donch_multi.config import DonchMultiConfig, ShortMode
from src.strategy.donch_multi.preprocessor import preprocess
from src.strategy.donch_multi.signal import generate_signals


@pytest.fixture
def config() -> DonchMultiConfig:
    return DonchMultiConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: DonchMultiConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: DonchMultiConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DonchMultiConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DonchMultiConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_shorts_require_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: 숏은 drawdown < hedge_threshold일 때만."""
        config = DonchMultiConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY에서 숏이 있다면 drawdown 조건 충족 구간에서만
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_shifted = df["drawdown"].shift(1)
            assert (dd_shifted[short_bars] < config.hedge_threshold).all()


class TestConsensusLogic:
    def test_unanimous_long(self, config: DonchMultiConfig) -> None:
        """3개 스케일 모두 long breakout → direction=1."""
        np.random.seed(42)
        n = 200
        # 강한 상승 트렌드
        close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.3
        high = close + 1.0
        low = close - 0.5
        open_ = close - 0.2
        volume = np.ones(n) * 5000.0
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # 충분한 warmup 이후 long direction 존재 확인
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        assert (post_warmup == 1).any()

    def test_unanimous_short(self) -> None:
        """3개 스케일 모두 short breakout → direction=-1 (FULL mode)."""
        config = DonchMultiConfig(short_mode=ShortMode.FULL)
        np.random.seed(42)
        n = 200
        # 강한 하락 트렌드
        close = 200 - np.arange(n) * 0.5 + np.random.randn(n) * 0.3
        high = close + 0.5
        low = close - 1.0
        open_ = close + 0.2
        volume = np.ones(n) * 5000.0
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        assert (post_warmup == -1).any()

    def test_entry_threshold_filters_weak_consensus(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """높은 threshold → 더 적은 진입."""
        config_low = DonchMultiConfig(entry_threshold=0.0)
        config_high = DonchMultiConfig(entry_threshold=0.67)
        df_low = preprocess(sample_ohlcv_df, config_low)
        df_high = preprocess(sample_ohlcv_df, config_high)
        sig_low = generate_signals(df_low, config_low)
        sig_high = generate_signals(df_high, config_high)
        active_low = (sig_low.direction != 0).sum()
        active_high = (sig_high.direction != 0).sum()
        assert active_low >= active_high

    def test_no_simultaneous_long_short(
        self, preprocessed_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        """동일 bar에서 long+short 동시 불가."""
        signals = generate_signals(preprocessed_df, config)
        assert not ((signals.direction == 1) & (signals.direction == -1)).any()

    def test_strength_sign_matches_direction(
        self, preprocessed_df: pd.DataFrame, config: DonchMultiConfig
    ) -> None:
        """strength 부호가 direction과 일치."""
        signals = generate_signals(preprocessed_df, config)
        active = signals.direction != 0
        if active.any():
            dir_sign = np.sign(signals.direction[active])
            str_sign = np.sign(signals.strength[active])
            # strength가 0이 아닌 경우만 비교
            nonzero = signals.strength[active] != 0
            if nonzero.any():
                assert (dir_sign[nonzero] == str_sign[nonzero]).all()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: DonchMultiConfig
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
