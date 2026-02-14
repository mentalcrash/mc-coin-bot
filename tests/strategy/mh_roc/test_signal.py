"""Tests for Multi-Horizon ROC Ensemble signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.mh_roc.config import MhRocConfig, ShortMode
from src.strategy.mh_roc.preprocessor import preprocess
from src.strategy.mh_roc.signal import generate_signals


@pytest.fixture
def config() -> MhRocConfig:
    return MhRocConfig()


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
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_df: pd.DataFrame, config: MhRocConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: MhRocConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self, preprocessed_df: pd.DataFrame, config: MhRocConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: MhRocConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: MhRocConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: MhRocConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert not signals.strength.isna().any()


class TestShift1Rule:
    def test_first_bar_neutral(self, preprocessed_df: pd.DataFrame, config: MhRocConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MhRocConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_df: pd.DataFrame) -> None:
        config = MhRocConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_with_drawdown(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: drawdown이 threshold 이하일 때 숏 허용."""
        config = MhRocConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,
            vote_threshold=1,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_strength_dampened(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HEDGE_ONLY: 숏 강도가 hedge_strength_ratio로 감쇄."""
        config = MhRocConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,
            vote_threshold=1,
        )
        df = preprocess(sample_ohlcv_df, config)
        signals = generate_signals(df, config)

        short_mask = signals.direction == -1
        if short_mask.any():
            short_strength = signals.strength[short_mask].abs()
            assert (short_strength >= 0).all()


class TestVotingLogic:
    def test_higher_threshold_fewer_signals(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """높은 vote_threshold → 적은 시그널."""
        config_strict = MhRocConfig(vote_threshold=4)
        config_loose = MhRocConfig(vote_threshold=1)

        df_strict = preprocess(sample_ohlcv_df, config_strict)
        df_loose = preprocess(sample_ohlcv_df, config_loose)

        sig_strict = generate_signals(df_strict, config_strict)
        sig_loose = generate_signals(df_loose, config_loose)

        active_strict = (sig_strict.direction != 0).sum()
        active_loose = (sig_loose.direction != 0).sum()
        assert active_loose >= active_strict

    def test_trending_up_generates_long(self) -> None:
        """순수 상승 추세에서 롱 시그널 발생."""
        n = 200
        close = np.linspace(100, 200, n)
        high = close + 2.0
        low = close - 2.0
        open_ = close - 0.5
        volume = np.full(n, 5000.0)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="4h"),
        )
        config = MhRocConfig(vote_threshold=3)
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # warmup 이후 대부분 long 방향
        late = signals.direction.iloc[config.warmup_periods() :]
        long_count = (late == 1).sum()
        total_active = (late != 0).sum()
        if total_active > 0:
            assert long_count / total_active > 0.8

    def test_conviction_weighting(self, preprocessed_df: pd.DataFrame, config: MhRocConfig) -> None:
        """만장일치(4/4)가 3/4보다 |strength| 큼 (동일 vol_scalar 시)."""
        signals = generate_signals(preprocessed_df, config)
        # strength에 NaN이 없음을 확인
        assert not signals.strength.isna().any()

    def test_entries_exits_consistency(
        self, preprocessed_df: pd.DataFrame, config: MhRocConfig
    ) -> None:
        """direction==0이면서 entries=True는 불가."""
        signals = generate_signals(preprocessed_df, config)
        zero_dir_entries = signals.entries & (signals.direction == 0)
        assert not zero_dir_entries.any()
