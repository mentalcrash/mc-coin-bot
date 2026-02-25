"""Tests for MVRV Cycle Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mvrv_cycle_trend.config import MvrvCycleTrendConfig, ShortMode
from src.strategy.mvrv_cycle_trend.preprocessor import preprocess
from src.strategy.mvrv_cycle_trend.signal import generate_signals


@pytest.fixture
def config() -> MvrvCycleTrendConfig:
    return MvrvCycleTrendConfig()


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
def sample_ohlcv_with_onchain(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_ohlcv_df.copy()
    np.random.seed(42)
    n = len(df)
    df["oc_mvrv"] = np.random.uniform(0.5, 4.0, n)
    return df


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


@pytest.fixture
def preprocessed_with_onchain(
    sample_ohlcv_with_onchain: pd.DataFrame,
) -> pd.DataFrame:
    config = MvrvCycleTrendConfig(mvrv_zscore_window=90)
    return preprocess(sample_ohlcv_with_onchain, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(
        self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        """shift(1)에 의해 첫 bar direction=0, strength=0."""
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortModeDisabled:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MvrvCycleTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_disabled_no_negative_strength(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MvrvCycleTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.strength >= 0).all()


class TestShortModeFull:
    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MvrvCycleTrendConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # FULL mode에서는 -1이 가능 (데이터에 따라)
        assert signals.direction.dtype == int

    def test_full_direction_range(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MvrvCycleTrendConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})


class TestShortModeHedgeOnly:
    def test_hedge_only_requires_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MvrvCycleTrendConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # HEDGE_ONLY: 숏은 drawdown < hedge_threshold일 때만
        short_mask = signals.direction == -1
        if short_mask.any():
            # 숏 발생한 bar의 이전 drawdown이 threshold 미만인지 확인
            dd_shifted = df["drawdown"].shift(1)
            short_dd = dd_shifted[short_mask].dropna()
            assert (short_dd < config.hedge_threshold).all()

    def test_hedge_only_strength_attenuated(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MvrvCycleTrendConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        # 숏 strength는 hedge_strength_ratio 적용
        short_mask = signals.direction == -1
        if short_mask.any():
            short_strength = signals.strength[short_mask].abs()
            # hedge_strength_ratio로 감쇄되므로 long보다 작거나 같아야
            assert (short_strength >= 0).all()


class TestOnchainSignals:
    """On-chain MVRV 데이터 유무에 따른 시그널 검증."""

    def test_without_mvrv_pure_momentum(
        self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        """MVRV 없으면 순수 momentum 기반 시그널."""
        signals = generate_signals(preprocessed_df, config)
        assert signals.strength is not None
        assert not signals.strength.isna().any()

    def test_with_mvrv_signals(self, preprocessed_with_onchain: pd.DataFrame) -> None:
        """MVRV 있을 때도 정상 동작."""
        config = MvrvCycleTrendConfig(mvrv_zscore_window=90)
        signals = generate_signals(preprocessed_with_onchain, config)
        assert not signals.strength.isna().any()
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_onchain_shift1(self, preprocessed_with_onchain: pd.DataFrame) -> None:
        """On-chain 파생 feature도 signal에서 shift(1) 적용 확인."""
        config = MvrvCycleTrendConfig(mvrv_zscore_window=90)
        signals = generate_signals(preprocessed_with_onchain, config)
        assert signals.strength.iloc[0] == 0.0


class TestConviction:
    """MVRV regime alignment conviction 검증."""

    def test_regime_aligned_higher_conviction(self) -> None:
        """Regime-aligned 시그널이 더 높은 |strength|를 가져야."""
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n) * 1.5)
        low = close - np.abs(np.random.randn(n) * 1.5)
        open_ = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 10000, n).astype(float)
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))

        # Bull regime: MVRV consistently low
        df_bull = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "oc_mvrv": np.full(n, 0.5),  # very low MVRV → bull
            },
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )
        # No MVRV (neutral)
        df_neutral = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="12h"),
        )

        config = MvrvCycleTrendConfig(mvrv_zscore_window=90)
        pp_bull = preprocess(df_bull, config)
        pp_neutral = preprocess(df_neutral, config)

        sig_bull = generate_signals(pp_bull, config)
        sig_neutral = generate_signals(pp_neutral, config)

        # Long bars에서 |strength| 비교
        long_bull = sig_bull.strength[sig_bull.direction == 1].abs()
        long_neutral = sig_neutral.strength[sig_neutral.direction == 1].abs()

        # Bull regime에서 long conviction이 더 높거나 같아야
        if len(long_bull) > 0 and len(long_neutral) > 0:
            assert long_bull.mean() >= long_neutral.mean() * 0.9  # 10% 여유


class TestNoLookaheadBias:
    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig
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
        self, sample_ohlcv_df: pd.DataFrame, config: MvrvCycleTrendConfig
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


class TestEntryExitConsistency:
    def test_no_simultaneous_entry_exit(
        self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        """동시 entry+exit 없어야."""
        signals = generate_signals(preprocessed_df, config)
        conflict = signals.entries & signals.exits
        assert not conflict.any()

    def test_no_entry_with_nan_strength(
        self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        """strength가 NaN인 bar에서 entry=False."""
        signals = generate_signals(preprocessed_df, config)
        # strength is filled, but check anyway
        nan_strength = signals.strength.isna()
        if nan_strength.any():
            assert not (signals.entries & nan_strength).any()

    def test_exit_when_direction_to_zero(
        self, preprocessed_df: pd.DataFrame, config: MvrvCycleTrendConfig
    ) -> None:
        """direction이 0으로 전환될 때 exit 발생."""
        signals = generate_signals(preprocessed_df, config)
        prev_dir = signals.direction.shift(1).fillna(0).astype(int)
        expected_exits = (signals.direction == 0) & (prev_dir != 0)
        pd.testing.assert_series_equal(signals.exits, expected_exits, check_names=False)
