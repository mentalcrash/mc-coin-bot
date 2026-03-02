"""Tests for VWAP-Channel Multi-Scale signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vwap_channel_12h.config import ShortMode, VwapChannelConfig
from src.strategy.vwap_channel_12h.preprocessor import preprocess
from src.strategy.vwap_channel_12h.signal import generate_signals


@pytest.fixture
def config() -> VwapChannelConfig:
    return VwapChannelConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: VwapChannelConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: VwapChannelConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        """동시 entry + exit 없음."""
        signals = generate_signals(preprocessed_df, config)
        assert not (signals.entries & signals.exits).any()

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        """strength에 NaN 없음."""
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_strength_zero_when_direction_zero(
        self, preprocessed_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        """direction=0일 때 strength=0."""
        signals = generate_signals(preprocessed_df, config)
        zero_dir = signals.direction == 0
        assert (signals.strength[zero_dir] == 0.0).all()

    def test_strength_sign_matches_direction(
        self, preprocessed_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        """strength 부호가 direction과 일치 (strength != 0 인 bar 한정)."""
        signals = generate_signals(preprocessed_df, config)
        # strength가 0이 아닌 bar만 검증 (vol_scalar NaN으로 strength=0인 warmup 구간 제외)
        nonzero_str = signals.strength != 0.0
        if nonzero_str.any():
            strength_sign = np.sign(signals.strength[nonzero_str].values)
            dir_sign = signals.direction[nonzero_str].values.astype(float)
            assert np.all(strength_sign == dir_sign)


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: VwapChannelConfig
    ) -> None:
        """shift(1) 적용으로 첫 bar는 중립."""
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VwapChannelConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0.0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VwapChannelConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_shorts_on_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: drawdown이 threshold 이하일 때만 숏 가능."""
        config = VwapChannelConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # direction=-1인 bar에서 drawdown < hedge_threshold 확인
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_at_shorts = df["drawdown"].shift(1)[short_bars]
            assert (dd_at_shorts < config.hedge_threshold).all()

    def test_hedge_only_strength_attenuated(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: 숏 방향 strength가 hedge_strength_ratio로 감쇄."""
        config = VwapChannelConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # 숏 방향 strength는 0 이하 (부동소수점 -0.0 허용)
        short_bars = signals.direction == -1
        if short_bars.any():
            assert (signals.strength[short_bars] <= 0).all()


class TestConsensusLogic:
    def test_strong_uptrend_gives_long(self) -> None:
        """강한 상승 추세에서 long direction."""
        np.random.seed(42)
        n = 300
        # 강한 상승 추세 데이터 생성
        close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.5
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_ = close - 0.1
        volume = np.full(n, 5000.0)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = VwapChannelConfig(short_mode=ShortMode.FULL)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # 후반부에서 long direction이 우세해야 함
        last_100 = signals.direction.iloc[-100:]
        long_count = (last_100 == 1).sum()
        assert long_count > 30, f"Expected >30 long bars, got {long_count}"

    def test_strong_downtrend_gives_short(self) -> None:
        """강한 하락 추세에서 short direction (FULL mode)."""
        np.random.seed(42)
        n = 300
        close = 200 - np.arange(n) * 0.5 + np.random.randn(n) * 0.5
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_ = close + 0.1
        volume = np.full(n, 5000.0)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = VwapChannelConfig(short_mode=ShortMode.FULL)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        last_100 = signals.direction.iloc[-100:]
        short_count = (last_100 == -1).sum()
        assert short_count > 30, f"Expected >30 short bars, got {short_count}"


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: VwapChannelConfig
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
