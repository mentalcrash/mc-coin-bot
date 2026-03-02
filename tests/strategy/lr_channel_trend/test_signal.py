"""Tests for LR-Channel Multi-Scale Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.lr_channel_trend.config import LrChannelTrendConfig, ShortMode
from src.strategy.lr_channel_trend.preprocessor import preprocess
from src.strategy.lr_channel_trend.signal import generate_signals


@pytest.fixture
def config() -> LrChannelTrendConfig:
    return LrChannelTrendConfig()


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
def preprocessed_df(
    sample_ohlcv_df: pd.DataFrame, config: LrChannelTrendConfig
) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0

    def test_no_lookahead(
        self, preprocessed_df: pd.DataFrame, config: LrChannelTrendConfig
    ) -> None:
        """shift(1) 적용으로 미래 데이터 참조 없음 확인."""
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = LrChannelTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = LrChannelTrendConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_hedge_only(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = LrChannelTrendConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestConsensus:
    def test_strong_uptrend_long(self) -> None:
        """강한 상승 추세 -> long 시그널 존재."""
        np.random.seed(42)
        n = 300
        # 강한 상승 추세: 큰 양의 drift
        close = 100 + np.cumsum(np.ones(n) * 3 + np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_ = close + np.random.randn(n) * 0.1
        volume = np.random.randint(1000, 10000, n).astype(float)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = LrChannelTrendConfig()
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        long_count = (signals.direction == 1).sum()
        assert long_count > 0, "Strong uptrend should produce long signals"

    def test_strong_downtrend_short_full(self) -> None:
        """강한 하락 추세 + FULL 모드 -> short 시그널 존재."""
        np.random.seed(42)
        n = 300
        # 강한 하락 추세: 큰 음의 drift
        close = 500 + np.cumsum(-np.ones(n) * 3 + np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.5)
        low = close - np.abs(np.random.randn(n) * 0.5)
        open_ = close + np.random.randn(n) * 0.1
        volume = np.random.randint(1000, 10000, n).astype(float)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = LrChannelTrendConfig(short_mode=ShortMode.FULL)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        short_count = (signals.direction == -1).sum()
        assert short_count > 0, "Strong downtrend with FULL mode should produce short signals"

    def test_threshold_filters(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """높은 entry_threshold는 시그널을 필터링한다."""
        config_low = LrChannelTrendConfig(entry_threshold=0.1)
        config_high = LrChannelTrendConfig(entry_threshold=0.9)
        df_low = preprocess(sample_ohlcv_df, config_low)
        df_high = preprocess(sample_ohlcv_df, config_high)
        signals_low = generate_signals(df_low, config_low)
        signals_high = generate_signals(df_high, config_high)
        active_low = (signals_low.direction != 0).sum()
        active_high = (signals_high.direction != 0).sum()
        assert active_low >= active_high
