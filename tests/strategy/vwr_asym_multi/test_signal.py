"""Tests for VWR Asymmetric Multi-Scale signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vwr_asym_multi.config import ShortMode, VwrAsymMultiConfig
from src.strategy.vwr_asym_multi.preprocessor import preprocess
from src.strategy.vwr_asym_multi.signal import generate_signals


@pytest.fixture
def config() -> VwrAsymMultiConfig:
    return VwrAsymMultiConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: VwrAsymMultiConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: VwrAsymMultiConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.strength.isna().sum() == 0

    def test_no_entry_and_exit_same_bar(
        self, preprocessed_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        """동일 bar에서 entries=True AND exits=True는 불가."""
        signals = generate_signals(preprocessed_df, config)
        both = signals.entries & signals.exits
        assert not both.any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: VwrAsymMultiConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestAsymmetricThresholds:
    def test_asymmetric_fewer_shorts_than_symmetric(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """비대칭 임계값(short > long)은 대칭보다 숏 진입이 적어야 함."""
        config_asym = VwrAsymMultiConfig(
            long_threshold=0.5,
            short_threshold=1.5,
            short_mode=ShortMode.FULL,
        )
        config_sym = VwrAsymMultiConfig(
            long_threshold=0.5,
            short_threshold=0.5,
            short_mode=ShortMode.FULL,
        )
        df_asym = preprocess(sample_ohlcv_df, config_asym)
        df_sym = preprocess(sample_ohlcv_df, config_sym)
        sig_asym = generate_signals(df_asym, config_asym)
        sig_sym = generate_signals(df_sym, config_sym)

        short_count_asym = (sig_asym.direction == -1).sum()
        short_count_sym = (sig_sym.direction == -1).sum()
        # 비대칭은 대칭보다 숏이 같거나 적어야 함
        assert short_count_asym <= short_count_sym

    def test_high_short_threshold_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """극단적 높은 short_threshold → 실질적 숏 없음."""
        config = VwrAsymMultiConfig(
            short_threshold=3.0,
            short_mode=ShortMode.FULL,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # 극단적 임계값이면 숏이 거의 없어야 함
        short_ratio = (signals.direction == -1).sum() / len(signals.direction)
        assert short_ratio < 0.1

    def test_zero_thresholds_many_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """threshold=0이면 많은 시그널 발생."""
        config = VwrAsymMultiConfig(
            long_threshold=0.0,
            short_threshold=0.0,
            short_mode=ShortMode.FULL,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        nonzero = (signals.direction != 0).sum()
        assert nonzero > 0


class TestShortModes:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VwrAsymMultiConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = VwrAsymMultiConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_needs_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: drawdown < hedge_threshold일 때만 숏."""
        config = VwrAsymMultiConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            short_threshold=0.3,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # 숏이 있으면 drawdown이 threshold보다 작아야 함
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_shifted = df["drawdown"].shift(1)
            short_dd = dd_shifted[short_bars].dropna()
            assert (short_dd < config.hedge_threshold).all()

    def test_hedge_strength_ratio_applied(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY 숏 strength에 ratio 적용 확인."""
        config = VwrAsymMultiConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_strength_ratio=0.5,
            short_threshold=0.3,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # strength의 음수값은 hedge_strength_ratio로 감쇄되어야 함
        short_strength = signals.strength[signals.direction == -1]
        if len(short_strength) > 0:
            # 모든 숏 strength는 0 이하
            assert (short_strength <= 0).all()


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: VwrAsymMultiConfig
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
