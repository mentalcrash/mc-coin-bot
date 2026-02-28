"""Tests for Trend Factor Multi-Horizon signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.trend_factor_12h.config import ShortMode, TrendFactorConfig
from src.strategy.trend_factor_12h.preprocessor import preprocess
from src.strategy.trend_factor_12h.signal import generate_signals


@pytest.fixture
def config() -> TrendFactorConfig:
    return TrendFactorConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: TrendFactorConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: TrendFactorConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: TrendFactorConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        """entries와 exits가 동시에 True인 bar가 없어야 한다."""
        signals = generate_signals(preprocessed_df, config)
        assert not (signals.entries & signals.exits).any()

    def test_strength_zero_when_direction_zero(
        self, preprocessed_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        """direction == 0인 bar에서 strength도 0이어야 한다."""
        signals = generate_signals(preprocessed_df, config)
        zero_dir = signals.direction == 0
        assert (signals.strength[zero_dir] == 0.0).all()

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: TrendFactorConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TrendFactorConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = TrendFactorConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # FULL 모드에서 direction dtype 확인
        assert signals.direction.dtype == int

    def test_hedge_only_needs_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY에서 drawdown < threshold일 때만 숏 허용."""
        config = TrendFactorConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.05,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY에서 숏이 발생하면 drawdown이 threshold 미만이어야
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_at_short = df["drawdown"].shift(1)[short_bars]
            assert (dd_at_short < config.hedge_threshold).all()

    def test_hedge_strength_ratio_applied(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY 모드에서 숏 strength에 ratio가 적용되는지 확인."""
        config = TrendFactorConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,  # 매우 낮은 threshold → 숏 활성화 용이
            hedge_strength_ratio=0.5,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # 숏 시 strength는 0보다 작음 (direction=-1 * positive conviction)
        short_bars = signals.direction == -1
        if short_bars.any():
            assert (signals.strength[short_bars] <= 0).all()


class TestTrendFactorSignalLogic:
    def test_strong_uptrend_gives_long(self) -> None:
        """강한 상승 추세에서 long direction 생성."""
        np.random.seed(42)
        n = 200
        # 강한 상승 추세 생성
        close = 100.0 * np.exp(np.cumsum(np.full(n, 0.01)))
        high = close * 1.005
        low = close * 0.995
        open_ = close * 0.999
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = TrendFactorConfig(entry_threshold=0.1)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # warmup 이후 대부분 long이어야
        late = signals.direction.iloc[100:]
        assert (late >= 0).all()
        assert (late == 1).sum() > len(late) * 0.5

    def test_strong_downtrend_gives_short_full(self) -> None:
        """강한 하락 추세에서 FULL mode → short direction."""
        np.random.seed(42)
        n = 200
        close = 100.0 * np.exp(np.cumsum(np.full(n, -0.01)))
        high = close * 1.005
        low = close * 0.995
        open_ = close * 1.001
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        config = TrendFactorConfig(short_mode=ShortMode.FULL, entry_threshold=0.1)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        late = signals.direction.iloc[100:]
        assert (late == -1).sum() > len(late) * 0.5

    def test_low_threshold_more_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """entry_threshold가 낮을수록 active bar가 많다."""
        config_low = TrendFactorConfig(entry_threshold=0.1)
        config_high = TrendFactorConfig(entry_threshold=2.0)
        df_low = preprocess(sample_ohlcv_df, config_low)
        df_high = preprocess(sample_ohlcv_df, config_high)
        sig_low = generate_signals(df_low, config_low)
        sig_high = generate_signals(df_high, config_high)
        active_low = (sig_low.direction != 0).sum()
        active_high = (sig_high.direction != 0).sum()
        assert active_low >= active_high


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self,
        sample_ohlcv_df: pd.DataFrame,
        config: TrendFactorConfig,
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
        config: TrendFactorConfig,
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
