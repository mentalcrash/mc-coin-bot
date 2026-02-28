"""Tests for Participation Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.participation_mom_12h.config import ParticipationMomConfig, ShortMode
from src.strategy.participation_mom_12h.preprocessor import preprocess
from src.strategy.participation_mom_12h.signal import generate_signals


@pytest.fixture
def config() -> ParticipationMomConfig:
    return ParticipationMomConfig()


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
def sample_ohlcv_with_tflow(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + tflow_intensity 포함 데이터."""
    df = sample_ohlcv_df.copy()
    np.random.seed(42)
    df["tflow_intensity"] = np.random.uniform(10, 200, len(df))
    return df


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: ParticipationMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


@pytest.fixture
def preprocessed_with_tflow(
    sample_ohlcv_with_tflow: pd.DataFrame, config: ParticipationMomConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_with_tflow, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        """동시 entry + exit 없음."""
        signals = generate_signals(preprocessed_df, config)
        assert not (signals.entries & signals.exits).any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: ParticipationMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortModes:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ParticipationMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ParticipationMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_requires_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = ParticipationMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY에서 short direction은 drawdown 조건에서만
        short_bars = signals.direction == -1
        if short_bars.any():
            dd = df["drawdown"].shift(1)
            assert (dd[short_bars] < config.hedge_threshold).all()

    def test_hedge_strength_ratio_applied(self, sample_ohlcv_with_tflow: pd.DataFrame) -> None:
        """HEDGE_ONLY 숏 strength가 FULL 대비 감쇄되는지 검증."""
        base_config = ParticipationMomConfig(
            short_mode=ShortMode.FULL,
            hedge_threshold=-0.01,
        )
        hedge_config = ParticipationMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,
            hedge_strength_ratio=0.5,
        )
        df_full = preprocess(sample_ohlcv_with_tflow, base_config)
        df_hedge = preprocess(sample_ohlcv_with_tflow, hedge_config)
        sig_full = generate_signals(df_full, base_config)
        sig_hedge = generate_signals(df_hedge, hedge_config)
        # HEDGE_ONLY의 숏 strength 절대값 평균이 FULL보다 작거나 같아야
        short_full = sig_full.direction == -1
        short_hedge = sig_hedge.direction == -1
        if short_full.any() and short_hedge.any():
            avg_full = sig_full.strength[short_full].abs().mean()
            avg_hedge = sig_hedge.strength[short_hedge].abs().mean()
            assert avg_hedge <= avg_full


class TestWithTradeFlow:
    """tflow_intensity가 있을 때 시그널 동작 검증."""

    def test_signals_with_tflow(
        self,
        preprocessed_with_tflow: pd.DataFrame,
        config: ParticipationMomConfig,
    ) -> None:
        signals = generate_signals(preprocessed_with_tflow, config)
        assert len(signals.entries) == len(preprocessed_with_tflow)

    def test_tflow_affects_signals(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_ohlcv_with_tflow: pd.DataFrame,
        config: ParticipationMomConfig,
    ) -> None:
        """tflow_intensity 유무에 따라 시그널이 달라야 함."""
        df_no_tflow = preprocess(sample_ohlcv_df, config)
        df_with_tflow = preprocess(sample_ohlcv_with_tflow, config)
        sig_no = generate_signals(df_no_tflow, config)
        sig_with = generate_signals(df_with_tflow, config)
        # 완전히 동일하지 않아야 함 (tflow가 필터 역할)
        assert not (sig_no.direction == sig_with.direction).all()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self,
        sample_ohlcv_df: pd.DataFrame,
        config: ParticipationMomConfig,
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
        self,
        sample_ohlcv_df: pd.DataFrame,
        config: ParticipationMomConfig,
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

    def test_truncation_invariance_with_tflow(
        self,
        sample_ohlcv_with_tflow: pd.DataFrame,
        config: ParticipationMomConfig,
    ) -> None:
        """tflow 포함 데이터에서도 truncation invariance 보장."""
        df_full = preprocess(sample_ohlcv_with_tflow, config)
        sig_full = generate_signals(df_full, config)
        cut = 50
        df_trunc = preprocess(sample_ohlcv_with_tflow.iloc[:-cut].copy(), config)
        sig_trunc = generate_signals(df_trunc, config)
        overlap = len(sig_trunc.direction)
        pd.testing.assert_series_equal(
            sig_full.direction.iloc[:overlap].reset_index(drop=True),
            sig_trunc.direction.reset_index(drop=True),
            check_names=False,
        )
