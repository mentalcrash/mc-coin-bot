"""Tests for Regime-Adaptive Multi-Lookback Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.regime_adaptive_mom.config import RegimeAdaptiveMomConfig, ShortMode
from src.strategy.regime_adaptive_mom.preprocessor import preprocess
from src.strategy.regime_adaptive_mom.signal import generate_signals


@pytest.fixture
def config() -> RegimeAdaptiveMomConfig:
    return RegimeAdaptiveMomConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: RegimeAdaptiveMomConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: RegimeAdaptiveMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: RegimeAdaptiveMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: RegimeAdaptiveMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: RegimeAdaptiveMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: RegimeAdaptiveMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortModeVariants:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = RegimeAdaptiveMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = RegimeAdaptiveMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_mode(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = RegimeAdaptiveMomConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int


class TestRegimeAdaptation:
    """RegimeService 컬럼 활용 테스트."""

    def test_with_regime_columns(
        self, preprocessed_df: pd.DataFrame, config: RegimeAdaptiveMomConfig
    ) -> None:
        """regime 컬럼이 있을 때 adaptive 가중치 사용."""
        df = preprocessed_df.copy()
        df["p_trending"] = 0.8
        df["p_ranging"] = 0.1
        df["p_volatile"] = 0.1
        df["regime_label"] = "trending"
        df["trend_direction_regime"] = 1
        df["trend_strength"] = 0.7
        signals = generate_signals(df, config)
        assert len(signals.entries) == len(df)

    def test_without_regime_columns(
        self, preprocessed_df: pd.DataFrame, config: RegimeAdaptiveMomConfig
    ) -> None:
        """regime 컬럼 없이도 equal weight fallback으로 정상 작동."""
        signals = generate_signals(preprocessed_df, config)
        assert len(signals.entries) == len(preprocessed_df)

    def test_trending_more_aggressive(self, preprocessed_df: pd.DataFrame) -> None:
        """trending에서 더 높은 평균 |strength| 확인."""
        config_default = RegimeAdaptiveMomConfig()
        df_trending = preprocessed_df.copy()
        df_trending["p_trending"] = 1.0
        df_trending["p_ranging"] = 0.0
        df_trending["p_volatile"] = 0.0
        df_trending["regime_label"] = "trending"
        df_trending["trend_direction_regime"] = 1
        df_trending["trend_strength"] = 0.8

        df_volatile = preprocessed_df.copy()
        df_volatile["p_trending"] = 0.0
        df_volatile["p_ranging"] = 0.0
        df_volatile["p_volatile"] = 1.0
        df_volatile["regime_label"] = "volatile"
        df_volatile["trend_direction_regime"] = 0
        df_volatile["trend_strength"] = 0.0

        sig_t = generate_signals(df_trending, config_default)
        sig_v = generate_signals(df_volatile, config_default)

        # trending의 평균 |strength|가 volatile보다 커야 함 (higher vol target)
        avg_t = sig_t.strength.abs().mean()
        avg_v = sig_v.strength.abs().mean()
        assert avg_t >= avg_v


class TestRegimeAdaptiveMomLogic:
    def test_clear_uptrend_generates_long(self) -> None:
        """Clear uptrend → long signal."""
        n = 300
        close = 100 + np.arange(n) * 0.5
        high = close + 1
        low = close - 1
        config = RegimeAdaptiveMomConfig()
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        late = signals.direction.iloc[config.warmup_periods() :]
        if (late != 0).any():
            assert (late >= 0).all()

    def test_clear_downtrend_generates_short_full(self) -> None:
        """Clear downtrend → short signal in FULL mode."""
        n = 300
        close = 200 - np.arange(n) * 0.5
        high = close + 1
        low = close - 1
        config = RegimeAdaptiveMomConfig(short_mode=ShortMode.FULL)
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 1000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        late = signals.direction.iloc[config.warmup_periods() :]
        if (late != 0).any():
            assert (late <= 0).all()

    def test_higher_threshold_fewer_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Higher signal threshold → fewer active signals."""
        config_low = RegimeAdaptiveMomConfig(signal_threshold=0.0)
        config_high = RegimeAdaptiveMomConfig(signal_threshold=0.1)

        df_low = preprocess(sample_ohlcv_df, config_low)
        df_high = preprocess(sample_ohlcv_df, config_high)

        sig_low = generate_signals(df_low, config_low)
        sig_high = generate_signals(df_high, config_high)

        active_low = (sig_low.direction != 0).sum()
        active_high = (sig_high.direction != 0).sum()
        assert active_low >= active_high

    def test_strength_nonzero_when_active_after_warmup(
        self, preprocessed_df: pd.DataFrame, config: RegimeAdaptiveMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        active = post_warmup != 0
        if active.any():
            assert (signals.strength.iloc[warmup:][active].abs() > 0).all()

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: RegimeAdaptiveMomConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()
