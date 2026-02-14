"""Tests for Carry-Conditional Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.carry_cond_mom.config import CarryCondMomConfig, ShortMode
from src.strategy.carry_cond_mom.preprocessor import preprocess
from src.strategy.carry_cond_mom.signal import generate_signals


@pytest.fixture
def config() -> CarryCondMomConfig:
    return CarryCondMomConfig()


@pytest.fixture
def sample_ohlcv_with_funding_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    funding_rate = np.random.uniform(-0.001, 0.001, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


@pytest.fixture
def preprocessed_df(
    sample_ohlcv_with_funding_df: pd.DataFrame, config: CarryCondMomConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_with_funding_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: CarryCondMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: CarryCondMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_with_funding_df: pd.DataFrame) -> None:
        config = CarryCondMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_with_funding_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_with_funding_df: pd.DataFrame) -> None:
        config = CarryCondMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_with_funding_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_with_drawdown(self, sample_ohlcv_with_funding_df: pd.DataFrame) -> None:
        config = CarryCondMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,
        )
        df = preprocess(sample_ohlcv_with_funding_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestAgreementLogic:
    def test_entries_exits_consistency(
        self, preprocessed_df: pd.DataFrame, config: CarryCondMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        zero_dir_entries = signals.entries & (signals.direction == 0)
        assert not zero_dir_entries.any()

    def test_agreement_boost_increases_strength(
        self, sample_ohlcv_with_funding_df: pd.DataFrame
    ) -> None:
        """agreement_boost가 높을수록 consensus 구간에서 stronger signals."""
        config_low = CarryCondMomConfig(agreement_boost=1.0)
        config_high = CarryCondMomConfig(agreement_boost=2.0)

        df_low = preprocess(sample_ohlcv_with_funding_df, config_low)
        df_high = preprocess(sample_ohlcv_with_funding_df, config_high)

        sig_low = generate_signals(df_low, config_low)
        sig_high = generate_signals(df_high, config_high)

        # Mean absolute strength comparison
        mean_low = sig_low.strength.abs().mean()
        mean_high = sig_high.strength.abs().mean()
        assert mean_high >= mean_low
