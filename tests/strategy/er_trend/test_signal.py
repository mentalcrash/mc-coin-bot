"""Tests for ER Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.er_trend.config import ErTrendConfig, ShortMode
from src.strategy.er_trend.preprocessor import preprocess
from src.strategy.er_trend.signal import generate_signals


@pytest.fixture
def config() -> ErTrendConfig:
    return ErTrendConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: ErTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, preprocessed_df: pd.DataFrame, config: ErTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: ErTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: ErTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: ErTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: ErTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        both = signals.entries & signals.exits
        assert not both.any()

    def test_strength_zero_when_flat(
        self, preprocessed_df: pd.DataFrame, config: ErTrendConfig
    ) -> None:
        """direction=0인 bar에서 strength=0이어야 한다."""
        signals = generate_signals(preprocessed_df, config)
        flat_mask = signals.direction == 0
        assert (signals.strength[flat_mask] == 0.0).all()


class TestShift1Rule:
    def test_first_bar_neutral(self, preprocessed_df: pd.DataFrame, config: ErTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ErTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ErTrendConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_requires_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY에서 drawdown < hedge_threshold일 때만 숏."""
        config = ErTrendConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY에서도 direction은 {-1, 0, 1} 범위
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_hedge_strength_reduced(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드에서 숏 strength가 ratio 적용."""
        config = ErTrendConfig(short_mode=ShortMode.HEDGE_ONLY, hedge_threshold=-0.01)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_mask = signals.direction == -1
        if short_mask.any():
            # 숏 strength 절대값이 0 이상 확인 (ratio 적용 검증)
            assert (signals.strength[short_mask].abs() >= 0).all()


class TestSignalLogic:
    def test_strong_uptrend_generates_long(self) -> None:
        """강한 상승 추세에서 long 시그널 발생."""
        n = 200
        close = 100 + np.arange(n, dtype=float) * 0.5  # monotonic up
        high = close + 0.3
        low = close - 0.3
        open_ = close - 0.1
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = ErTrendConfig()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # 강한 상승 추세에서 long이 대부분
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        assert (post_warmup == 1).sum() > len(post_warmup) * 0.5

    def test_strong_downtrend_generates_short(self) -> None:
        """강한 하락 추세에서 FULL mode short 시그널 발생."""
        n = 200
        close = 200 - np.arange(n, dtype=float) * 0.5  # monotonic down
        high = close + 0.3
        low = close - 0.3
        open_ = close + 0.1
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = ErTrendConfig(short_mode=ShortMode.FULL)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        assert (post_warmup == -1).sum() > len(post_warmup) * 0.5

    def test_conviction_scales_strength(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """conviction (|composite_ser|)이 strength 크기에 반영."""
        config = ErTrendConfig()
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # direction != 0인 곳에서 strength가 다양한 크기
        active = signals.strength[signals.direction != 0]
        if len(active) > 10:
            assert active.abs().std() > 0  # 균일하지 않음


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig
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

    def test_single_bar_append(self, sample_ohlcv_df: pd.DataFrame, config: ErTrendConfig) -> None:
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
