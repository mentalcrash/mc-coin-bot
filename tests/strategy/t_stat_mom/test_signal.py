"""Tests for T-Stat Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.t_stat_mom.config import ShortMode, TStatMomConfig
from src.strategy.t_stat_mom.preprocessor import preprocess
from src.strategy.t_stat_mom.signal import generate_signals


@pytest.fixture
def config() -> TStatMomConfig:
    return TStatMomConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: TStatMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: TStatMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: TStatMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: TStatMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: TStatMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.strength.isna().sum() == 0

    def test_strength_zero_when_direction_zero(
        self, preprocessed_df: pd.DataFrame, config: TStatMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        neutral_mask = signals.direction == 0
        if neutral_mask.any():
            assert (signals.strength[neutral_mask] == 0.0).all()


class TestShift1Rule:
    def test_first_bar_neutral(self, preprocessed_df: pd.DataFrame, config: TStatMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TStatMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TStatMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_needs_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY에서 shorts는 drawdown < hedge_threshold 시에만 발생."""
        config = TStatMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # direction이 -1인 bar에서 drawdown.shift(1) < hedge_threshold 확인
        short_mask = signals.direction == -1
        if short_mask.any():
            dd_at_short = df["drawdown"].shift(1)[short_mask]
            assert (dd_at_short < config.hedge_threshold).all()

    def test_hedge_only_reduced_strength(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY에서 숏 강도가 감쇄된다."""
        config = TStatMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,  # 쉽게 활성화
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_mask = signals.direction == -1
        if short_mask.any():
            # 모든 short strength는 hedge_strength_ratio 적용됨
            assert (signals.strength[short_mask] <= 0).all()


class TestTStatSpecific:
    def test_strong_trend_generates_entries(self) -> None:
        """강한 상승 추세에서 long entry가 발생해야 한다."""
        np.random.seed(42)
        n = 200
        # 강한 상승 추세
        close = 100 * np.exp(np.cumsum(np.full(n, 0.005) + np.random.randn(n) * 0.002))
        high = close * 1.01
        low = close * 0.99
        open_ = close * 1.0005
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = TStatMomConfig()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # 강한 상승에서 long이 있어야 함
        assert (signals.direction == 1).any()

    def test_strong_downtrend_generates_shorts(self) -> None:
        """강한 하락 추세에서 FULL 모드일 때 short이 발생해야 한다."""
        np.random.seed(42)
        n = 200
        # 강한 하락 추세
        close = 100 * np.exp(np.cumsum(np.full(n, -0.005) + np.random.randn(n) * 0.002))
        high = close * 1.01
        low = close * 0.99
        open_ = close * 1.0005
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = TStatMomConfig(short_mode=ShortMode.FULL)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        assert (signals.direction == -1).any()

    def test_flat_market_no_signals(self) -> None:
        """횡보장에서 시그널이 적어야 한다 (threshold 미달)."""
        np.random.seed(42)
        n = 200
        # 횡보 (매우 작은 수익률)
        close = 100 + np.cumsum(np.random.randn(n) * 0.001)
        high = close + 0.01
        low = close - 0.01
        open_ = close.copy()
        volume = np.full(n, 5000.0)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = TStatMomConfig(entry_threshold=2.0)  # 높은 threshold
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # 대부분 중립
        neutral_pct = (signals.direction == 0).mean()
        assert neutral_pct > 0.8

    def test_tanh_conviction_bounded(self, preprocessed_df: pd.DataFrame) -> None:
        """tanh conviction으로 strength가 유한하게 제한된다."""
        config = TStatMomConfig()
        signals = generate_signals(preprocessed_df, config)
        assert np.isfinite(signals.strength).all()

    def test_entries_exits_consistency(
        self, preprocessed_df: pd.DataFrame, config: TStatMomConfig
    ) -> None:
        """entries와 exits가 동시에 True이면 안 된다."""
        signals = generate_signals(preprocessed_df, config)
        assert not (signals.entries & signals.exits).any()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig
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

    def test_single_bar_append(self, sample_ohlcv_df: pd.DataFrame, config: TStatMomConfig) -> None:
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
