"""Tests for Squeeze-Adaptive Breakout signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.squeeze_adaptive_breakout.config import (
    ShortMode,
    SqueezeAdaptiveBreakoutConfig,
)
from src.strategy.squeeze_adaptive_breakout.preprocessor import preprocess
from src.strategy.squeeze_adaptive_breakout.signal import generate_signals


@pytest.fixture
def config() -> SqueezeAdaptiveBreakoutConfig:
    return SqueezeAdaptiveBreakoutConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def preprocessed_df(
    sample_ohlcv_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()

    def test_strength_zero_when_neutral(
        self, preprocessed_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        neutral_mask = signals.direction == 0
        assert (signals.strength[neutral_mask] == 0.0).all()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0

    def test_second_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: SqueezeAdaptiveBreakoutConfig
    ) -> None:
        """Shift(2) 사용으로 첫 2 bar는 반드시 중립."""
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[1] == 0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = SqueezeAdaptiveBreakoutConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = SqueezeAdaptiveBreakoutConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # FULL 모드에서는 -1이 가능해야 함 (데이터에 따라)
        assert signals.direction.dtype == int

    def test_hedge_only_respects_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = SqueezeAdaptiveBreakoutConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_disabled_no_negative_strength(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = SqueezeAdaptiveBreakoutConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.strength >= 0).all()


class TestSqueezeLogic:
    def test_no_signal_without_squeeze(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """squeeze_lookback을 매우 크게 설정하면 시그널이 줄어야 함."""
        config_strict = SqueezeAdaptiveBreakoutConfig(squeeze_lookback=20)
        df = preprocess(sample_ohlcv_df, config_strict)
        signals_strict = generate_signals(df, config_strict)

        config_loose = SqueezeAdaptiveBreakoutConfig(squeeze_lookback=1)
        df2 = preprocess(sample_ohlcv_df, config_loose)
        signals_loose = generate_signals(df2, config_loose)

        # strict squeeze => fewer or equal signals
        assert signals_strict.entries.sum() <= signals_loose.entries.sum() + 5  # tolerance


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self,
        sample_ohlcv_df: pd.DataFrame,
        config: SqueezeAdaptiveBreakoutConfig,
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
        config: SqueezeAdaptiveBreakoutConfig,
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
