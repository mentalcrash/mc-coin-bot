"""Tests for Regime-Gated Multi-Factor MR signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.regime_mf_mr.config import RegimeMfMrConfig, ShortMode
from src.strategy.regime_mf_mr.preprocessor import preprocess
from src.strategy.regime_mf_mr.signal import generate_signals


@pytest.fixture
def config() -> RegimeMfMrConfig:
    return RegimeMfMrConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: RegimeMfMrConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: RegimeMfMrConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: RegimeMfMrConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: RegimeMfMrConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: RegimeMfMrConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: RegimeMfMrConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = RegimeMfMrConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = RegimeMfMrConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_mode(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = RegimeMfMrConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY: shorts only when drawdown exceeds threshold
        assert signals.direction.dtype == int


class TestRegimeAdaptation:
    """RegimeService 컬럼 활용 테스트."""

    def test_with_regime_columns(
        self, preprocessed_df: pd.DataFrame, config: RegimeMfMrConfig
    ) -> None:
        """regime 컬럼이 있을 때 adaptive vol_target 사용."""
        df = preprocessed_df.copy()
        df["p_trending"] = 0.1
        df["p_ranging"] = 0.8
        df["p_volatile"] = 0.1
        df["regime_label"] = "ranging"
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)

    def test_without_regime_columns(
        self, preprocessed_df: pd.DataFrame, config: RegimeMfMrConfig
    ) -> None:
        """regime 컬럼 없이도 fallback으로 정상 작동."""
        signals = generate_signals(preprocessed_df, config)
        assert len(signals.entries) == len(preprocessed_df)

    def test_ranging_more_active(self, preprocessed_df: pd.DataFrame) -> None:
        """ranging에서 MR 시그널 활성화 확인."""
        config = RegimeMfMrConfig()

        # Ranging regime: gate open
        df_ranging = preprocessed_df.copy()
        df_ranging["p_trending"] = 0.1
        df_ranging["p_ranging"] = 0.8
        df_ranging["p_volatile"] = 0.1
        df_ranging["regime_label"] = "ranging"

        # Trending regime: gate closed (vol_target = 0)
        df_trending = preprocessed_df.copy()
        df_trending["p_trending"] = 0.8
        df_trending["p_ranging"] = 0.1
        df_trending["p_volatile"] = 0.1
        df_trending["regime_label"] = "trending"

        sig_r = generate_signals(df_ranging, config)
        sig_t = generate_signals(df_trending, config)

        # ranging에서의 평균 |strength|가 trending보다 크거나 같아야 함
        avg_r = sig_r.strength.abs().mean()
        avg_t = sig_t.strength.abs().mean()
        assert avg_r >= avg_t

    def test_regime_gate_blocks_trending(self, preprocessed_df: pd.DataFrame) -> None:
        """trending 레짐에서 regime_gate가 시그널 차단."""
        config = RegimeMfMrConfig(regime_gate_threshold=0.5)

        df = preprocessed_df.copy()
        df["p_trending"] = 0.8
        df["p_ranging"] = 0.1  # < 0.5 threshold
        df["p_volatile"] = 0.1
        df["regime_label"] = "trending"

        signals = generate_signals(df, config)
        # regime gate가 닫혀서 시그널이 없어야 함
        assert (signals.direction == 0).all()
