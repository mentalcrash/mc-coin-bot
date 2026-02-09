"""Tests for HMM Regime signal generator."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.hmm_regime.config import HMMRegimeConfig, ShortMode
from src.strategy.hmm_regime.preprocessor import preprocess
from src.strategy.hmm_regime.signal import generate_signals
from src.strategy.types import Direction


@pytest.fixture
def default_config() -> HMMRegimeConfig:
    """Small config for fast tests."""
    return HMMRegimeConfig(
        n_states=3,
        n_iter=50,
        min_train_window=100,
        retrain_interval=50,
        vol_window=20,
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Sample OHLCV DataFrame (300 days) with regime structure."""
    np.random.seed(42)
    n = 300

    returns = np.zeros(n)
    returns[:100] = np.random.randn(100) * 0.02 + 0.003
    returns[100:200] = np.random.randn(100) * 0.025 - 0.004
    returns[200:] = np.random.randn(100) * 0.02 + 0.003

    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )
    return df


@pytest.fixture
def preprocessed_df(
    sample_ohlcv_df: pd.DataFrame,
    default_config: HMMRegimeConfig,
) -> pd.DataFrame:
    """Preprocessed DataFrame."""
    return preprocess(sample_ohlcv_df, default_config)


class TestSignalStructure:
    """Signal output structure tests."""

    def test_signal_output_structure(self, preprocessed_df: pd.DataFrame) -> None:
        """Signal output has entries, exits, direction, strength fields."""
        signals = generate_signals(preprocessed_df)

        assert hasattr(signals, "entries")
        assert hasattr(signals, "exits")
        assert hasattr(signals, "direction")
        assert hasattr(signals, "strength")

        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int

        assert len(signals.entries) == len(preprocessed_df)
        assert len(signals.exits) == len(preprocessed_df)
        assert len(signals.direction) == len(preprocessed_df)
        assert len(signals.strength) == len(preprocessed_df)

    def test_direction_values(self, preprocessed_df: pd.DataFrame) -> None:
        """Direction values are subset of {-1, 0, 1}."""
        config = HMMRegimeConfig(
            n_states=3,
            n_iter=50,
            min_train_window=100,
            retrain_interval=50,
            short_mode=ShortMode.FULL,
        )
        signals = generate_signals(preprocessed_df, config)

        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )


class TestRegimeSignals:
    """Regime-based signal generation tests."""

    def test_bull_regime_generates_long(self) -> None:
        """Bull regime (1) should generate long signals."""
        n = 50
        df = pd.DataFrame(
            {
                "regime": [1] * n,
                "regime_prob": [0.9] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        signals = generate_signals(df)

        # After shift(1), from bar 1 onward, direction should be LONG
        assert (signals.direction.iloc[1:] == Direction.LONG).all()

    def test_bear_regime_generates_short_full_mode(self) -> None:
        """Bear regime (-1) with FULL short_mode should generate short signals."""
        n = 50
        df = pd.DataFrame(
            {
                "regime": [-1] * n,
                "regime_prob": [0.9] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        config = HMMRegimeConfig(
            n_states=3,
            n_iter=50,
            min_train_window=100,
            retrain_interval=50,
            short_mode=ShortMode.FULL,
        )
        signals = generate_signals(df, config)

        # After shift(1), from bar 1 onward, direction should be SHORT
        assert (signals.direction.iloc[1:] == Direction.SHORT).all()

    def test_short_mode_disabled(self, preprocessed_df: pd.DataFrame) -> None:
        """DISABLED short_mode: no -1 in direction."""
        config = HMMRegimeConfig(
            n_states=3,
            n_iter=50,
            min_train_window=100,
            retrain_interval=50,
            short_mode=ShortMode.DISABLED,
        )
        signals = generate_signals(preprocessed_df, config)

        # No SHORT direction when disabled
        assert (signals.direction >= 0).all()
        assert (signals.strength >= 0).all()

    def test_unknown_regime_flat(self) -> None:
        """Unknown regime (-1 in raw regime) or Sideways (0) should be FLAT."""
        n = 50
        # Test with unknown regime (-1 in regime column, meaning not yet classified)
        df_unknown = pd.DataFrame(
            {
                "regime": [-1] * n,
                "regime_prob": [0.0] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        signals = generate_signals(df_unknown)
        # Unknown regime maps to Bear direction_raw=-1, but DISABLED mode suppresses
        # Default is DISABLED, so shorts become neutral
        assert (signals.direction == Direction.NEUTRAL).all()

        # Test with sideways regime (0)
        df_sideways = pd.DataFrame(
            {
                "regime": [0] * n,
                "regime_prob": [0.5] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        signals_sideways = generate_signals(df_sideways)
        assert (signals_sideways.direction == Direction.NEUTRAL).all()


class TestShift1Rule:
    """Shift(1) rule (lookahead bias prevention) tests."""

    def test_shift1_no_lookahead(self) -> None:
        """Signal at bar t uses regime at bar t-1 (shift(1) applied)."""
        n = 10
        # Regime switches from 0 to 1 at bar 5
        regime = [0] * 5 + [1] * 5
        df = pd.DataFrame(
            {
                "regime": regime,
                "regime_prob": [0.9] * n,
                "vol_scalar": [1.0] * n,
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        signals = generate_signals(df)

        # Bar 0: shift(1) -> NaN -> NEUTRAL
        assert signals.direction.iloc[0] == Direction.NEUTRAL

        # Bar 5: should still use regime at bar 4 (which is 0=Sideways) -> NEUTRAL
        assert signals.direction.iloc[5] == Direction.NEUTRAL

        # Bar 6: should use regime at bar 5 (which is 1=Bull) -> LONG
        assert signals.direction.iloc[6] == Direction.LONG
