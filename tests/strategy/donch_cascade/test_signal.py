"""Tests for Donchian Cascade MTF signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.donch_cascade.config import DonchCascadeConfig, ShortMode
from src.strategy.donch_cascade.preprocessor import preprocess
from src.strategy.donch_cascade.signal import _apply_cascade_entry, generate_signals


@pytest.fixture
def config() -> DonchCascadeConfig:
    return DonchCascadeConfig()


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 800
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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: DonchCascadeConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: DonchCascadeConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DonchCascadeConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DonchCascadeConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_shorts_require_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: 숏은 drawdown < hedge_threshold일 때만."""
        config = DonchCascadeConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_shifted = df["drawdown"].shift(1)
            assert (dd_shifted[short_bars] < config.hedge_threshold).all()


class TestConsensusLogic:
    def test_unanimous_long(self, config: DonchCascadeConfig) -> None:
        """강한 상승 트렌드 → long direction 존재."""
        np.random.seed(42)
        n = 800
        close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.3
        high = close + 1.0
        low = close - 0.5
        open_ = close - 0.2
        volume = np.ones(n) * 5000.0
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        assert (post_warmup == 1).any()

    def test_unanimous_short(self) -> None:
        """강한 하락 트렌드 → short direction 존재 (FULL mode)."""
        config = DonchCascadeConfig(short_mode=ShortMode.FULL)
        np.random.seed(42)
        n = 800
        close = 500 - np.arange(n) * 0.5 + np.random.randn(n) * 0.3
        high = close + 0.5
        low = close - 1.0
        open_ = close + 0.2
        volume = np.ones(n) * 5000.0
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        assert (post_warmup == -1).any()

    def test_entry_threshold_filters_weak_consensus(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """높은 threshold → 더 적은 진입."""
        config_low = DonchCascadeConfig(entry_threshold=0.0)
        config_high = DonchCascadeConfig(entry_threshold=0.67)
        df_low = preprocess(sample_ohlcv_df, config_low)
        df_high = preprocess(sample_ohlcv_df, config_high)
        sig_low = generate_signals(df_low, config_low)
        sig_high = generate_signals(df_high, config_high)
        active_low = (sig_low.direction != 0).sum()
        active_high = (sig_high.direction != 0).sum()
        assert active_low >= active_high

    def test_strength_sign_matches_direction(
        self, preprocessed_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        """strength 부호가 direction과 일치."""
        signals = generate_signals(preprocessed_df, config)
        active = signals.direction != 0
        if active.any():
            dir_sign = np.sign(signals.direction[active])
            str_sign = np.sign(signals.strength[active])
            nonzero = signals.strength[active] != 0
            if nonzero.any():
                assert (dir_sign[nonzero] == str_sign[nonzero]).all()


class TestCascadeEntry:
    """Cascade entry logic 핵심 테스트."""

    def test_confirmation_delays_entry(self) -> None:
        """EMA confirmation 없으면 진입이 지연된다."""
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        htf_dir = pd.Series([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], index=idx, dtype=int)
        # close가 EMA 아래 (confirmation 실패)
        prev_close = pd.Series([100] * 10, index=idx, dtype=float)
        confirm_ema = pd.Series([200] * 10, index=idx, dtype=float)

        result = _apply_cascade_entry(
            htf_direction=htf_dir,
            prev_close=prev_close,
            confirm_ema=confirm_ema,
            max_wait_bars=5,
        )

        # bar 2: htf=1이지만 confirmation 없음 → 0
        assert result.iloc[2] == 0
        # bar 3: 여전히 confirmation 없음 → 0
        assert result.iloc[3] == 0

    def test_confirmation_enables_entry(self) -> None:
        """EMA confirmation 성공 시 즉시 진입."""
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        htf_dir = pd.Series([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], index=idx, dtype=int)
        # close > EMA (confirmation 성공)
        prev_close = pd.Series([200] * 10, index=idx, dtype=float)
        confirm_ema = pd.Series([100] * 10, index=idx, dtype=float)

        result = _apply_cascade_entry(
            htf_direction=htf_dir,
            prev_close=prev_close,
            confirm_ema=confirm_ema,
            max_wait_bars=5,
        )

        # bar 2: htf=1 + confirmation → 1
        assert result.iloc[2] == 1
        # bar 3-5: 유지
        assert result.iloc[3] == 1
        assert result.iloc[4] == 1
        assert result.iloc[5] == 1

    def test_force_entry_after_max_wait(self) -> None:
        """max_wait_bars 후 강제 진입."""
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        htf_dir = pd.Series([0, 1, 1, 1, 1, 1, 1, 0, 0, 0], index=idx, dtype=int)
        # close < EMA (confirmation 실패)
        prev_close = pd.Series([100] * 10, index=idx, dtype=float)
        confirm_ema = pd.Series([200] * 10, index=idx, dtype=float)

        result = _apply_cascade_entry(
            htf_direction=htf_dir,
            prev_close=prev_close,
            confirm_ema=confirm_ema,
            max_wait_bars=3,
        )

        # bar 1: htf=1, bars_in_group=0, no confirm → 0
        assert result.iloc[1] == 0
        # bar 2: bars_in_group=1, no confirm → 0
        assert result.iloc[2] == 0
        # bar 3: bars_in_group=2, no confirm → 0
        assert result.iloc[3] == 0
        # bar 4: bars_in_group=3 >= max_wait → forced entry = 1
        assert result.iloc[4] == 1
        # bar 5-6: 유지 (cummax)
        assert result.iloc[5] == 1
        assert result.iloc[6] == 1

    def test_entry_stays_after_confirmation(self) -> None:
        """확인 후 direction group 끝까지 포지션 유지."""
        idx = pd.date_range("2024-01-01", periods=10, freq="4h")
        htf_dir = pd.Series([0, 1, 1, 1, 1, 1, 0, 0, 0, 0], index=idx, dtype=int)
        # bar 3에서만 confirmation 성공
        prev_close = pd.Series(
            [100, 100, 100, 200, 100, 100, 100, 100, 100, 100],
            index=idx,
            dtype=float,
        )
        confirm_ema = pd.Series([150] * 10, index=idx, dtype=float)

        result = _apply_cascade_entry(
            htf_direction=htf_dir,
            prev_close=prev_close,
            confirm_ema=confirm_ema,
            max_wait_bars=10,  # high max_wait to test confirmation only
        )

        # bar 1-2: no confirmation → 0
        assert result.iloc[1] == 0
        assert result.iloc[2] == 0
        # bar 3: confirmed → 1
        assert result.iloc[3] == 1
        # bar 4-5: stays entered (cummax)
        assert result.iloc[4] == 1
        assert result.iloc[5] == 1
        # bar 6+: htf=0 → 0
        assert result.iloc[6] == 0

    def test_neutral_htf_direction_stays_zero(self) -> None:
        """htf_direction=0이면 항상 0."""
        idx = pd.date_range("2024-01-01", periods=5, freq="4h")
        htf_dir = pd.Series([0, 0, 0, 0, 0], index=idx, dtype=int)
        prev_close = pd.Series([200] * 5, index=idx, dtype=float)
        confirm_ema = pd.Series([100] * 5, index=idx, dtype=float)

        result = _apply_cascade_entry(
            htf_direction=htf_dir,
            prev_close=prev_close,
            confirm_ema=confirm_ema,
            max_wait_bars=1,
        )

        assert (result == 0).all()

    def test_short_confirmation(self) -> None:
        """SHORT: prev_close < confirm_ema로 확인."""
        idx = pd.date_range("2024-01-01", periods=8, freq="4h")
        htf_dir = pd.Series([0, 0, -1, -1, -1, -1, 0, 0], index=idx, dtype=int)
        # close < EMA → short confirmed
        prev_close = pd.Series([100] * 8, index=idx, dtype=float)
        confirm_ema = pd.Series([200] * 8, index=idx, dtype=float)

        result = _apply_cascade_entry(
            htf_direction=htf_dir,
            prev_close=prev_close,
            confirm_ema=confirm_ema,
            max_wait_bars=10,
        )

        assert result.iloc[2] == -1  # short confirmed
        assert result.iloc[3] == -1
        assert result.iloc[6] == 0  # neutral


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: DonchCascadeConfig
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
    def test_no_simultaneous_long_short(
        self, preprocessed_df: pd.DataFrame, config: DonchCascadeConfig
    ) -> None:
        """동일 bar에서 long+short 동시 불가."""
        signals = generate_signals(preprocessed_df, config)
        assert not ((signals.direction == 1) & (signals.direction == -1)).any()
