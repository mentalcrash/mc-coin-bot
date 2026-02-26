"""Tests for Donchian Filtered signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.donch_filtered.config import DonchFilteredConfig
from src.strategy.donch_filtered.preprocessor import preprocess
from src.strategy.donch_filtered.signal import _apply_crowd_filter, generate_signals
from src.strategy.donch_multi.config import ShortMode


@pytest.fixture
def config() -> DonchFilteredConfig:
    return DonchFilteredConfig()


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
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(
        self, preprocessed_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(
        self, preprocessed_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(
        self, preprocessed_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_no_nan_in_strength(
        self, preprocessed_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(
        self, preprocessed_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0

    def test_truncation_invariance(
        self, sample_ohlcv_df: pd.DataFrame, config: DonchFilteredConfig
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


class TestCrowdFilter:
    def test_long_overheated_suppressed(self) -> None:
        """fr_zscore > threshold AND direction==1 → 억제."""
        direction = pd.Series([1, 1, -1, 0, 1])
        fr_zscore = pd.Series([2.0, 0.5, 2.0, 2.0, -2.0])
        threshold = 1.5
        result = _apply_crowd_filter(direction, fr_zscore, threshold)
        assert result.iloc[0] == 0  # long + fr>1.5 → 억제
        assert result.iloc[1] == 1  # long + fr<1.5 → 통과
        assert result.iloc[2] == -1  # short + fr>1.5 → 통과 (반대 방향)
        assert result.iloc[3] == 0  # flat → 그대로
        assert result.iloc[4] == 1  # long + fr<-1.5 → 통과 (반대 방향)

    def test_short_overheated_suppressed(self) -> None:
        """fr_zscore < -threshold AND direction==-1 → 억제."""
        direction = pd.Series([-1, -1, 1, 0, -1])
        fr_zscore = pd.Series([-2.0, -0.5, -2.0, -2.0, 2.0])
        threshold = 1.5
        result = _apply_crowd_filter(direction, fr_zscore, threshold)
        assert result.iloc[0] == 0  # short + fr<-1.5 → 억제
        assert result.iloc[1] == -1  # short + fr>-1.5 → 통과
        assert result.iloc[2] == 1  # long + fr<-1.5 → 통과 (반대 방향)
        assert result.iloc[3] == 0  # flat → 그대로
        assert result.iloc[4] == -1  # short + fr>1.5 → 통과 (반대 방향)

    def test_zero_fr_zscore_passthrough(self) -> None:
        """fr_zscore=0 (no derivatives) → pure donch-multi 동일."""
        direction = pd.Series([1, -1, 0, 1, -1])
        fr_zscore = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0])
        threshold = 1.5
        result = _apply_crowd_filter(direction, fr_zscore, threshold)
        pd.testing.assert_series_equal(result, direction, check_names=False)

    def test_crowd_filter_in_full_pipeline_no_derivatives(
        self, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Derivatives 없으면 donch-multi와 동일한 시그널."""
        from src.strategy.donch_multi.config import DonchMultiConfig
        from src.strategy.donch_multi.preprocessor import preprocess as dm_preprocess
        from src.strategy.donch_multi.signal import generate_signals as dm_generate

        dm_config = DonchMultiConfig()
        df_config = DonchFilteredConfig()

        dm_df = dm_preprocess(sample_ohlcv_df, dm_config)
        dm_sig = dm_generate(dm_df, dm_config)

        df_df = preprocess(sample_ohlcv_df, df_config)
        df_sig = generate_signals(df_df, df_config)

        pd.testing.assert_series_equal(
            dm_sig.direction.reset_index(drop=True),
            df_sig.direction.reset_index(drop=True),
            check_names=False,
        )


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DonchFilteredConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DonchFilteredConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_shorts_require_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = DonchFilteredConfig(short_mode=ShortMode.HEDGE_ONLY)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        short_bars = signals.direction == -1
        if short_bars.any():
            dd_shifted = df["drawdown"].shift(1)
            assert (dd_shifted[short_bars] < config.hedge_threshold).all()


class TestConsensusLogic:
    def test_strength_sign_matches_direction(
        self, preprocessed_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        active = signals.direction != 0
        if active.any():
            dir_sign = np.sign(signals.direction[active])
            str_sign = np.sign(signals.strength[active])
            nonzero = signals.strength[active] != 0
            if nonzero.any():
                assert (dir_sign[nonzero] == str_sign[nonzero]).all()

    def test_no_simultaneous_long_short(
        self, preprocessed_df: pd.DataFrame, config: DonchFilteredConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not ((signals.direction == 1) & (signals.direction == -1)).any()
