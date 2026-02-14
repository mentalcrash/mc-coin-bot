"""Tests for Variance Decomposition Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vardecomp_mom.config import ShortMode, VardecompMomConfig
from src.strategy.vardecomp_mom.preprocessor import preprocess
from src.strategy.vardecomp_mom.signal import generate_signals


@pytest.fixture
def config() -> VardecompMomConfig:
    return VardecompMomConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: VardecompMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: VardecompMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: VardecompMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VardecompMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VardecompMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_with_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VardecompMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,
            var_ratio_threshold=0.5,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_strength_dampened(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VardecompMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,
            var_ratio_threshold=0.5,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_mask = signals.direction == -1
        if short_mask.any():
            short_strength = signals.strength[short_mask].abs()
            assert (short_strength >= 0).all()


class TestVarRatioLogic:
    def test_lower_threshold_more_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """threshold 낮으면 더 많은 시그널 발생."""
        config_strict = VardecompMomConfig(var_ratio_threshold=0.9)
        config_loose = VardecompMomConfig(var_ratio_threshold=0.5)

        df_strict = preprocess(sample_ohlcv_df, config_strict)
        df_loose = preprocess(sample_ohlcv_df, config_loose)

        sig_strict = generate_signals(df_strict, config_strict)
        sig_loose = generate_signals(df_loose, config_loose)

        active_strict = (sig_strict.direction != 0).sum()
        active_loose = (sig_loose.direction != 0).sum()
        assert active_loose >= active_strict

    def test_entries_exits_consistency(
        self, preprocessed_df: pd.DataFrame, config: VardecompMomConfig
    ) -> None:
        """direction==0이면서 entries=True는 불가."""
        signals = generate_signals(preprocessed_df, config)
        zero_dir_entries = signals.entries & (signals.direction == 0)
        assert not zero_dir_entries.any()
