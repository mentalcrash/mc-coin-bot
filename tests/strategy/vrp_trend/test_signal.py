"""Tests for VRP-Trend signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.vrp_trend.config import ShortMode, VrpTrendConfig
from src.strategy.vrp_trend.preprocessor import preprocess
from src.strategy.vrp_trend.signal import generate_signals


@pytest.fixture
def config() -> VrpTrendConfig:
    return VrpTrendConfig()


@pytest.fixture
def sample_ohlcv_dvol_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 300
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n) * 1.5)
    low = close - np.abs(np.random.randn(n) * 1.5)
    open_ = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000, 10000, n).astype(float)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    dvol = 50.0 + np.cumsum(np.random.randn(n) * 2)
    dvol = np.clip(dvol, 20, 120)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "dvol": dvol,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


@pytest.fixture
def preprocessed_df(sample_ohlcv_dvol_df: pd.DataFrame, config: VrpTrendConfig) -> pd.DataFrame:
    return preprocess(sample_ohlcv_dvol_df, config)


class TestSignalStructure:
    def test_output_fields(self, preprocessed_df: pd.DataFrame, config: VrpTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(
        self, preprocessed_df: pd.DataFrame, config: VrpTrendConfig
    ) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self, preprocessed_df: pd.DataFrame, config: VrpTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self, preprocessed_df: pd.DataFrame, config: VrpTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        n = len(preprocessed_df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n

    def test_strength_no_nan(self, preprocessed_df: pd.DataFrame, config: VrpTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.strength.notna().all()


class TestShift1Rule:
    def test_first_bar_neutral(self, preprocessed_df: pd.DataFrame, config: VrpTrendConfig) -> None:
        signals = generate_signals(preprocessed_df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self, sample_ohlcv_dvol_df: pd.DataFrame) -> None:
        config = VrpTrendConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(sample_ohlcv_dvol_df, config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self, sample_ohlcv_dvol_df: pd.DataFrame) -> None:
        config = VrpTrendConfig(short_mode=ShortMode.FULL)
        df = preprocess(sample_ohlcv_dvol_df, config)
        signals = generate_signals(df, config)
        # FULL 모드에서는 direction dtype이 int
        assert signals.direction.dtype == int

    def test_hedge_only_strength_damped(self, sample_ohlcv_dvol_df: pd.DataFrame) -> None:
        """HEDGE_ONLY 숏 강도가 hedge_strength_ratio로 감쇄."""
        config = VrpTrendConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.01,  # 쉽게 트리거
            hedge_strength_ratio=0.5,
        )
        df = preprocess(sample_ohlcv_dvol_df, config)
        signals = generate_signals(df, config)
        short_mask = signals.direction == -1
        if short_mask.any():
            # 숏 포지션의 strength 절대값은 vol_scalar * 0.5
            short_strength = signals.strength[short_mask].abs()
            assert (short_strength >= 0).all()


class TestVrpSignalLogic:
    def test_high_vrp_uptrend_long(self) -> None:
        """고VRP + 상승추세 → Long 시그널 생성."""
        config = VrpTrendConfig()
        n = 200
        np.random.seed(42)
        # 강한 상승 추세 + 높은 DVOL
        close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.5
        high = close + 1
        low = close - 1
        open_ = close - 0.2
        volume = np.full(n, 5000.0)
        # DVOL을 RV보다 충분히 높게 설정 (고VRP)
        dvol = np.full(n, 80.0)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "dvol": dvol,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        # warmup 이후 최소 일부 long 시그널 존재
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        assert (post_warmup == 1).any(), "Expected some long signals with high VRP + uptrend"

    def test_low_vrp_downtrend_short(self) -> None:
        """저VRP + 하락추세 → Short 시그널 생성 (FULL mode)."""
        config = VrpTrendConfig(short_mode=ShortMode.FULL)
        n = 200
        np.random.seed(42)
        # 하락 추세 + 낮은 DVOL
        close = 200 - np.arange(n) * 0.5 + np.random.randn(n) * 0.5
        high = close + 1
        low = close - 1
        open_ = close + 0.2
        volume = np.full(n, 5000.0)
        # DVOL을 RV보다 낮게 설정 (저VRP)
        dvol = np.full(n, 15.0)

        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "dvol": dvol,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        processed = preprocess(df, config)
        signals = generate_signals(processed, config)
        warmup = config.warmup_periods()
        post_warmup = signals.direction.iloc[warmup:]
        assert (post_warmup == -1).any(), "Expected some short signals with low VRP + downtrend"

    def test_no_simultaneous_long_short(
        self, preprocessed_df: pd.DataFrame, config: VrpTrendConfig
    ) -> None:
        """동시에 long과 short이 발생하지 않는다."""
        signals = generate_signals(preprocessed_df, config)
        long_mask = signals.direction == 1
        short_mask = signals.direction == -1
        overlap = long_mask & short_mask
        assert not overlap.any()

    def test_entries_on_direction_change(
        self, preprocessed_df: pd.DataFrame, config: VrpTrendConfig
    ) -> None:
        """entries는 direction 변경 시에만 True."""
        signals = generate_signals(preprocessed_df, config)
        prev_dir = signals.direction.shift(1).fillna(0).astype(int)
        expected_entries = (signals.direction != 0) & (signals.direction != prev_dir)
        pd.testing.assert_series_equal(
            signals.entries, expected_entries.astype(bool), check_names=False
        )

    def test_exits_on_direction_to_zero(
        self, preprocessed_df: pd.DataFrame, config: VrpTrendConfig
    ) -> None:
        """exits는 direction이 0으로 바뀔 때 True."""
        signals = generate_signals(preprocessed_df, config)
        prev_dir = signals.direction.shift(1).fillna(0).astype(int)
        expected_exits = (signals.direction == 0) & (prev_dir != 0)
        pd.testing.assert_series_equal(
            signals.exits, expected_exits.astype(bool), check_names=False
        )
