"""Tests for ML Derivatives Regime signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.ml_deriv_regime.config import MlDerivRegimeConfig, ShortMode
from src.strategy.ml_deriv_regime.preprocessor import preprocess
from src.strategy.ml_deriv_regime.signal import generate_signals


@pytest.fixture
def config() -> MlDerivRegimeConfig:
    # Use small training window for faster tests
    return MlDerivRegimeConfig(training_window=60)


@pytest.fixture
def sample_ohlcv_with_funding_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
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
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def preprocessed_df(
    sample_ohlcv_with_funding_df: pd.DataFrame, config: MlDerivRegimeConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_with_funding_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: MlDerivRegimeConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_with_funding_df: pd.DataFrame) -> None:
        config = MlDerivRegimeConfig(short_mode=ShortMode.DISABLED, training_window=60)
        df = preprocess(sample_ohlcv_with_funding_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_with_funding_df: pd.DataFrame) -> None:
        config = MlDerivRegimeConfig(short_mode=ShortMode.FULL, training_window=60)
        df = preprocess(sample_ohlcv_with_funding_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestIncrementalMode:
    def test_predict_last_only(
        self, preprocessed_df: pd.DataFrame, config: MlDerivRegimeConfig
    ) -> None:
        """predict_last_only=True가 에러 없이 동작."""
        signals = generate_signals(preprocessed_df, config, predict_last_only=True)
        assert len(signals.entries) == len(preprocessed_df)
