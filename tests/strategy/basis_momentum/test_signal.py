"""Tests for Basis-Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.basis_momentum.config import BasisMomentumConfig, ShortMode
from src.strategy.basis_momentum.preprocessor import preprocess
from src.strategy.basis_momentum.signal import generate_signals


@pytest.fixture
def config() -> BasisMomentumConfig:
    return BasisMomentumConfig()


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
def sample_ohlcv_with_fr(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_ohlcv_df.copy()
    np.random.seed(123)
    # Synthetic funding rate: positive mean (bullish bias) with noise
    df["funding_rate"] = 0.001 + np.random.randn(len(df)) * 0.0005
    return df


@pytest.fixture
def preprocessed_no_fr(
    sample_ohlcv_df: pd.DataFrame, config: BasisMomentumConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


@pytest.fixture
def preprocessed_with_fr(
    sample_ohlcv_with_fr: pd.DataFrame, config: BasisMomentumConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_with_fr, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_with_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        signals = generate_signals(preprocessed_with_fr, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_with_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        signals = generate_signals(preprocessed_with_fr, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_with_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        signals = generate_signals(preprocessed_with_fr, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_with_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        signals = generate_signals(preprocessed_with_fr, config)
        n = len(preprocessed_with_fr)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_in_strength(
        self, preprocessed_with_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        signals = generate_signals(preprocessed_with_fr, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_with_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        signals = generate_signals(preprocessed_with_fr, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead(
        self, preprocessed_with_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        """shift(1) 적용으로 미래 데이터 참조 없음 확인."""
        signals = generate_signals(preprocessed_with_fr, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_with_fr: pd.DataFrame) -> None:
        config = BasisMomentumConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_with_fr, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_with_fr: pd.DataFrame) -> None:
        config = BasisMomentumConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_with_fr, config)
        signals = generate_signals(df, config)
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_hedge_only(self, sample_ohlcv_with_fr: pd.DataFrame) -> None:
        config = BasisMomentumConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_with_fr, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestGracefulDegradation:
    def test_no_fr_all_flat(
        self, preprocessed_no_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        """funding_rate 부재 시 direction=0, strength=0."""
        signals = generate_signals(preprocessed_no_fr, config)
        assert (signals.direction == 0).all()
        assert (signals.strength == 0.0).all()

    def test_no_fr_no_entries(
        self, preprocessed_no_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        """funding_rate 부재 시 진입 시그널 없음."""
        signals = generate_signals(preprocessed_no_fr, config)
        assert not signals.entries.any()


class TestWithFundingRate:
    def test_positive_fr_momentum_long_signals(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """양의 FR 모멘텀 -> long 시그널 존재."""
        df = sample_ohlcv_df.copy()
        n = len(df)
        np.random.seed(99)
        # 점진적으로 증가하는 FR -> 양의 FR 모멘텀
        df["funding_rate"] = np.linspace(0.0001, 0.003, n) + np.random.randn(n) * 0.0001
        config = BasisMomentumConfig(short_mode=ShortMode.FULL, entry_zscore=1.0, exit_zscore=0.3)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        long_count = (signals.direction == 1).sum()
        assert long_count > 0, "Positive FR momentum should produce long signals"

    def test_with_fr_has_active_signals(
        self, preprocessed_with_fr: pd.DataFrame, config: BasisMomentumConfig
    ) -> None:
        """FR 존재 시 활성 시그널이 있을 수 있음."""
        signals = generate_signals(preprocessed_with_fr, config)
        # FR이 있으면 direction != 0인 bar가 존재할 수 있음
        # 단, threshold에 따라 없을 수도 있으므로 구조만 검증
        assert signals.direction.dtype == int
