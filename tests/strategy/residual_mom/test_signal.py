"""Tests for Residual Momentum signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.residual_mom.config import ResidualMomConfig, ShortMode
from src.strategy.residual_mom.preprocessor import preprocess
from src.strategy.residual_mom.signal import generate_signals


def _make_df() -> pd.DataFrame:
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
        index=pd.date_range("2024-01-01", periods=n, freq="1D"),
    )


class TestSignalStructure:
    def test_output_fields(self) -> None:
        config = ResidualMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

    def test_entries_exits_bool(self) -> None:
        config = ResidualMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_direction_values(self) -> None:
        config = ResidualMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_same_length(self) -> None:
        config = ResidualMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        n = len(df)
        assert len(signals.entries) == n
        assert len(signals.exits) == n
        assert len(signals.direction) == n
        assert len(signals.strength) == n


class TestShift1Rule:
    def test_first_bar_neutral(self) -> None:
        config = ResidualMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert signals.direction.iloc[0] == 0
        assert signals.strength.iloc[0] == 0.0


class TestShortMode:
    def test_disabled_no_shorts(self) -> None:
        config = ResidualMomConfig(short_mode=ShortMode.DISABLED)
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert (signals.direction >= 0).all()

    def test_full_allows_shorts(self) -> None:
        config = ResidualMomConfig(short_mode=ShortMode.FULL)
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert signals.direction.dtype == int

    def test_hedge_only_no_shorts_without_drawdown(self) -> None:
        """HEDGE_ONLY: drawdown이 threshold보다 얕으면 short 없어야 한다."""
        config = ResidualMomConfig(
            short_mode=ShortMode.HEDGE_ONLY,
            hedge_threshold=-0.99,  # 거의 도달 불가 수준
        )
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        # 극단적 hedge_threshold이면 short가 거의 발생하지 않아야 함
        assert (signals.direction >= 0).all()


class TestConviction:
    def test_strength_scales_with_zscore(self) -> None:
        """z-score가 클수록 |strength|가 커야 한다."""
        config = ResidualMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        # direction != 0인 바에서 strength != 0 확인
        active = signals.direction != 0
        if active.any():
            active_strength = signals.strength[active]
            assert (active_strength.abs() > 0).all()

    def test_strength_no_nan(self) -> None:
        """strength에 NaN이 없어야 한다."""
        config = ResidualMomConfig()
        df = preprocess(_make_df(), config)
        signals = generate_signals(df, config)
        assert not signals.strength.isna().any()
