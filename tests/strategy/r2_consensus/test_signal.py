"""Tests for R2 Consensus Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.r2_consensus.config import R2ConsensusConfig, ShortMode
from src.strategy.r2_consensus.preprocessor import preprocess
from src.strategy.r2_consensus.signal import generate_signals


@pytest.fixture
def config() -> R2ConsensusConfig:
    return R2ConsensusConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: R2ConsensusConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: R2ConsensusConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_entries_exits_mutually_exclusive(
        self, preprocessed_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        """entries와 exits가 동일 bar에서 동시 발생하지 않음."""
        signals = generate_signals(preprocessed_df, config)
        assert not (signals.entries & signals.exits).any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = R2ConsensusConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = R2ConsensusConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_shorts_during_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드에서는 drawdown 시에만 숏 허용."""
        config = R2ConsensusConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # 숏 진입 시 이전 봉 drawdown이 hedge_threshold 미만이어야 함
        short_bars = signals.direction == -1
        if short_bars.any():
            dd = df["drawdown"].shift(1)
            assert (dd[short_bars] < config.hedge_threshold).all()

    def test_hedge_only_strength_reduced(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드에서 숏 strength에 hedge_strength_ratio 적용 확인."""
        config = R2ConsensusConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_strength_ratio=0.5,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # 숏 strength는 hedge_strength_ratio가 적용되어야 함
        short_bars = signals.direction == -1
        if short_bars.any():
            # 숏 strength의 절대값이 vol_scalar보다 작아야 함 (ratio < 1)
            vol_scalar = df["vol_scalar"].shift(1)
            short_strength = signals.strength[short_bars].abs()
            short_vol = vol_scalar[short_bars].abs()
            # 최소한 하나라도 검증 가능하면 ratio 적용 확인
            valid = short_vol.dropna()
            if len(valid) > 0:
                assert (short_strength[valid.index] <= short_vol[valid.index] + 1e-10).all()


class TestConsensusLogic:
    def test_unanimous_long_generates_signal(self) -> None:
        """3개 스케일 모두 long일 때 시그널 생성."""
        config = R2ConsensusConfig(entry_threshold=0.34)
        n = 200
        np.random.seed(42)
        # 강한 상승 추세: 모든 스케일에서 R^2 높고 slope 양수
        close = 100.0 + np.arange(n, dtype=float) * 0.5
        close += np.random.randn(n) * 0.1  # 약간의 noise
        high = close + 1.0
        low = close - 1.0
        open_ = close - 0.1
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # 충분한 warmup 이후 long 시그널 존재해야 함
        late = signals.direction.iloc[140:]
        assert (late == 1).any(), "Should have long signals in strong uptrend"

    def test_no_consensus_no_signal(self) -> None:
        """consensus가 threshold 미만이면 direction=0."""
        config = R2ConsensusConfig(entry_threshold=0.99)  # 극히 높은 threshold
        np.random.seed(42)
        n = 300
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)
        open_ = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 10000, n).astype(float)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # entry_threshold=0.99 -> consensus 절대값이 0.99 이상이어야 진입
        # consensus는 최대 1.0이므로 매우 드물거나 없음
        non_zero = (signals.direction != 0).sum()
        assert non_zero <= len(df) * 0.1, "Very high threshold should suppress most signals"

    def test_direction_sign_matches_consensus(
        self, preprocessed_df: pd.DataFrame, config: R2ConsensusConfig
    ) -> None:
        """direction 부호가 consensus 부호와 일치."""
        signals = generate_signals(preprocessed_df, config)
        # direction이 양수인 곳의 strength도 양수 (또는 0)
        long_bars = signals.direction == 1
        if long_bars.any():
            assert (signals.strength[long_bars] >= 0).all()
        # direction이 음수인 곳의 strength도 음수 (또는 0)
        short_bars = signals.direction == -1
        if short_bars.any():
            assert (signals.strength[short_bars] <= 0).all()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: R2ConsensusConfig
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
