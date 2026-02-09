"""Tests for HMM Regime preprocessor."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.hmm_regime.config import HMMRegimeConfig
from src.strategy.hmm_regime.preprocessor import (
    _map_state_to_regime,
    preprocess,
)


@pytest.fixture
def sample_config() -> HMMRegimeConfig:
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
    """Sample OHLCV DataFrame (300 days) with clear regime structure.

    Creates synthetic data with bull/bear phases for HMM training:
    - Days 0-99: Bull phase (uptrend)
    - Days 100-199: Bear phase (downtrend)
    - Days 200-299: Bull phase (uptrend again)
    """
    np.random.seed(42)
    n = 300

    # Create regime-structured returns
    returns = np.zeros(n)
    # Bull phase: positive drift
    returns[:100] = np.random.randn(100) * 0.02 + 0.003
    # Bear phase: negative drift
    returns[100:200] = np.random.randn(100) * 0.025 - 0.004
    # Bull phase again
    returns[200:] = np.random.randn(100) * 0.02 + 0.003

    # Build price series from returns
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


class TestPreprocessColumns:
    """Preprocess output column tests."""

    def test_preprocess_returns_all_columns(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HMMRegimeConfig,
    ) -> None:
        """Preprocess output contains all required columns."""
        result = preprocess(sample_ohlcv_df, sample_config)

        expected_cols = [
            "regime",
            "regime_prob",
            "vol_scalar",
            "rolling_vol",
            "atr",
            "drawdown",
            "returns",
            "realized_vol",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_missing_columns_raises(self, sample_config: HMMRegimeConfig) -> None:
        """Missing required columns raises ValueError."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, sample_config)


class TestRegimeClassification:
    """HMM regime classification tests."""

    def test_regime_values(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HMMRegimeConfig,
    ) -> None:
        """Regime values are in {-1, 0, 1} for classified bars."""
        result = preprocess(sample_ohlcv_df, sample_config)
        regime_series = result["regime"]

        # Classified bars (regime >= 0) should only be -1, 0, or 1
        classified = regime_series[regime_series >= 0]
        if len(classified) > 0:
            assert set(classified.unique()).issubset({-1, 0, 1})

    def test_unknown_before_min_train(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HMMRegimeConfig,
    ) -> None:
        """Regime is -1 (unknown) for bars before min_train_window."""
        result = preprocess(sample_ohlcv_df, sample_config)
        regime_series = result["regime"]

        # All bars before min_train_window should be unknown (-1)
        early_regimes = regime_series.iloc[: sample_config.min_train_window]
        assert (early_regimes == -1).all()

    def test_regime_prob_range(
        self,
        sample_ohlcv_df: pd.DataFrame,
        sample_config: HMMRegimeConfig,
    ) -> None:
        """Regime probability is in [0, 1] for classified bars."""
        result = preprocess(sample_ohlcv_df, sample_config)
        regime_prob = result["regime_prob"]
        regime = result["regime"]

        # For classified bars (regime >= 0), probability should be in [0, 1]
        classified_mask = regime >= 0
        if classified_mask.any():
            probs = regime_prob[classified_mask]
            assert (probs >= 0).all()
            assert (probs <= 1).all()


class TestMapStateToRegime:
    """_map_state_to_regime unit tests."""

    def test_map_state_to_regime(self) -> None:
        """Test state-to-regime mapping with known means."""
        # 3 states: means = [-0.01, 0.001, 0.02]
        # sorted_states = [0, 1, 2] (ascending by mean)
        # State 0 (mean=-0.01) -> Bear (-1)
        # State 1 (mean=0.001) -> Sideways (0)
        # State 2 (mean=0.02) -> Bull (1)
        means = np.array([-0.01, 0.001, 0.02])

        assert _map_state_to_regime(means, state=0, n_states=3) == -1  # Bear
        assert _map_state_to_regime(means, state=1, n_states=3) == 0  # Sideways
        assert _map_state_to_regime(means, state=2, n_states=3) == 1  # Bull

    def test_map_state_to_regime_2_states(self) -> None:
        """Test 2-state mapping: highest=Bull, lowest=Bear."""
        means = np.array([-0.005, 0.01])

        assert _map_state_to_regime(means, state=0, n_states=2) == -1  # Bear
        assert _map_state_to_regime(means, state=1, n_states=2) == 1  # Bull

    def test_map_state_to_regime_reversed_means(self) -> None:
        """Test mapping when state ordering differs from index ordering."""
        # State 0 has high mean, State 2 has low mean
        means = np.array([0.03, 0.0, -0.02])

        assert _map_state_to_regime(means, state=0, n_states=3) == 1  # Bull (highest mean)
        assert _map_state_to_regime(means, state=1, n_states=3) == 0  # Sideways
        assert _map_state_to_regime(means, state=2, n_states=3) == -1  # Bear (lowest mean)

    def test_map_state_to_regime_5_states(self) -> None:
        """Test 5-state mapping: lowest=Bear, highest=Bull, rest=Sideways."""
        means = np.array([-0.03, -0.01, 0.001, 0.01, 0.05])

        assert _map_state_to_regime(means, state=0, n_states=5) == -1  # Bear
        assert _map_state_to_regime(means, state=1, n_states=5) == 0  # Sideways
        assert _map_state_to_regime(means, state=2, n_states=5) == 0  # Sideways
        assert _map_state_to_regime(means, state=3, n_states=5) == 0  # Sideways
        assert _map_state_to_regime(means, state=4, n_states=5) == 1  # Bull
