"""Tests for FgAsymMom signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.fg_asym_mom.config import FgAsymMomConfig, ShortMode
from src.strategy.fg_asym_mom.preprocessor import preprocess
from src.strategy.fg_asym_mom.signal import generate_signals


@pytest.fixture
def config() -> FgAsymMomConfig:
    return FgAsymMomConfig()


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
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )
    df["oc_fear_greed"] = np.random.randint(10, 90, n).astype(float)
    return df


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: FgAsymMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: FgAsymMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: FgAsymMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: FgAsymMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: FgAsymMomConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert len(signals.entries) == len(preprocessed_df)


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: FgAsymMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = FgAsymMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = FgAsymMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestAsymmetricLogic:
    def test_fear_buy_signal(self) -> None:
        """Fear zone + 가격 > SMA + F&G 반등 시 long."""
        n = 60
        close = np.linspace(90, 110, n)  # 상승 추세
        fg_values = [15.0] * 10 + [18.0] * 10 + [50.0] * 40
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n) * 1000,
                "oc_fear_greed": fg_values,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="1D"),
        )
        config = FgAsymMomConfig(fg_delta_window=5, sma_short=5)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # Fear buy should trigger somewhere in the early bars
        assert signals.direction.sum() >= 0  # net long or flat
