"""Tests for VRP-Regime Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vrp_regime_trend.config import ShortMode, VrpRegimeTrendConfig
from src.strategy.vrp_regime_trend.preprocessor import preprocess
from src.strategy.vrp_regime_trend.signal import generate_signals


@pytest.fixture
def config() -> VrpRegimeTrendConfig:
    return VrpRegimeTrendConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="8h"),
    )


@pytest.fixture
def sample_ohlcv_with_dvol(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_ohlcv_df.copy()
    np.random.seed(123)
    df["opt_dvol"] = np.random.uniform(40, 80, len(df))
    return df


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: VrpRegimeTrendConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


@pytest.fixture
def preprocessed_df_with_dvol(
    sample_ohlcv_with_dvol: pd.DataFrame, config: VrpRegimeTrendConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_with_dvol, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: VrpRegimeTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_with_dvol: pd.DataFrame) -> None:
        config = VrpRegimeTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_with_dvol, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_with_dvol: pd.DataFrame) -> None:
        config = VrpRegimeTrendConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_with_dvol, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_conditional_shorts(self, sample_ohlcv_with_dvol: pd.DataFrame) -> None:
        config = VrpRegimeTrendConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_with_dvol, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY: short 발생 시 drawdown < hedge_threshold 조건 필수
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_shifted = df["drawdown"].shift(1)
            assert (dd_shifted[short_bars] < config.hedge_threshold).all()

    def test_hedge_only_strength_ratio(self, sample_ohlcv_with_dvol: pd.DataFrame) -> None:
        config = VrpRegimeTrendConfig(short_mode=ShortMode.HEDGE_ONLY, hedge_strength_ratio=0.5)
        df = preprocess(sample_ohlcv_with_dvol, config)
        signals = generate_signals(df, config)
        # Strength NaN 없음
        assert not signals.strength.isna().any()


class TestSignalWithDvol:
    """DVOL 존재 시 실제 VRP 기반 시그널 테스트."""

    def test_generates_nonzero_signals(
        self,
        preprocessed_df_with_dvol: pd.DataFrame,
        config: VrpRegimeTrendConfig,
    ) -> None:
        """DVOL 존재 시 시그널이 생성됨."""
        signals = generate_signals(preprocessed_df_with_dvol, config)
        assert (signals.direction != 0).any()

    def test_strength_no_nan(
        self,
        preprocessed_df_with_dvol: pd.DataFrame,
        config: VrpRegimeTrendConfig,
    ) -> None:
        signals = generate_signals(preprocessed_df_with_dvol, config)
        assert not signals.strength.isna().any()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_with_dvol: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        """마지막 50 bar 제거해도 이전 시그널 동일."""
        df_full = preprocess(sample_ohlcv_with_dvol, config)
        sig_full = generate_signals(df_full, config)
        cut = 50
        df_trunc = preprocess(sample_ohlcv_with_dvol.iloc[:-cut].copy(), config)
        sig_trunc = generate_signals(df_trunc, config)
        overlap = len(sig_trunc.direction)
        pd.testing.assert_series_equal(
            sig_full.direction.iloc[:overlap].reset_index(drop=True),
            sig_trunc.direction.reset_index(drop=True),
            check_names=False,
        )

    def test_single_bar_append(
        self, sample_ohlcv_with_dvol: pd.DataFrame, config: VrpRegimeTrendConfig
    ) -> None:
        """1 bar 추가 시 마지막 bar만 변경 가능."""
        n = len(sample_ohlcv_with_dvol)
        df_prev = preprocess(sample_ohlcv_with_dvol.iloc[: n - 1].copy(), config)
        sig_prev = generate_signals(df_prev, config)
        df_full = preprocess(sample_ohlcv_with_dvol, config)
        sig_full = generate_signals(df_full, config)
        pd.testing.assert_series_equal(
            sig_full.direction.iloc[:-1].reset_index(drop=True),
            sig_prev.direction.reset_index(drop=True),
            check_names=False,
        )
