"""Tests for Return Streak Persistence preprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.streak_persistence.config import StreakPersistenceConfig
from src.strategy.streak_persistence.preprocessor import preprocess


@pytest.fixture
def config() -> StreakPersistenceConfig:
    return StreakPersistenceConfig()


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


class TestPreprocess:
    def test_output_columns(
        self, sample_ohlcv_df: pd.DataFrame, config: StreakPersistenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        required = {
            "returns",
            "realized_vol",
            "vol_scalar",
            "positive_streak",
            "negative_streak",
            "momentum",
            "drawdown",
            "atr",
        }
        assert required.issubset(set(result.columns))

    def test_same_length(
        self, sample_ohlcv_df: pd.DataFrame, config: StreakPersistenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert len(result) == len(sample_ohlcv_df)

    def test_immutability(
        self, sample_ohlcv_df: pd.DataFrame, config: StreakPersistenceConfig
    ) -> None:
        original = sample_ohlcv_df.copy()
        preprocess(sample_ohlcv_df, config)
        pd.testing.assert_frame_equal(sample_ohlcv_df, original)

    def test_missing_columns(self, config: StreakPersistenceConfig) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess(df, config)

    def test_vol_scalar_positive(
        self, sample_ohlcv_df: pd.DataFrame, config: StreakPersistenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        valid = result["vol_scalar"].dropna()
        assert (valid > 0).all()

    def test_streak_non_negative(
        self, sample_ohlcv_df: pd.DataFrame, config: StreakPersistenceConfig
    ) -> None:
        result = preprocess(sample_ohlcv_df, config)
        assert (result["positive_streak"] >= 0).all()
        assert (result["negative_streak"] >= 0).all()

    def test_streaks_mutually_exclusive(
        self, sample_ohlcv_df: pd.DataFrame, config: StreakPersistenceConfig
    ) -> None:
        """Positive and negative streaks cannot both be > 0 at same bar."""
        result = preprocess(sample_ohlcv_df, config)
        both_active = (result["positive_streak"] > 0) & (result["negative_streak"] > 0)
        assert not both_active.any()

    def test_known_streak_pattern(self) -> None:
        """Test with a known pattern: 5 up, 3 down."""
        n = 20
        close = np.array([100.0] * n)
        # Bars 1-5: up (return > 0)
        for i in range(1, 6):
            close[i] = close[i - 1] + 2.0
        # Bars 6-8: down (return < 0)
        for i in range(6, 9):
            close[i] = close[i - 1] - 2.0
        # Bars 9+: flat
        for i in range(9, n):
            close[i] = close[i - 1]

        high = close + 1.0
        low = close - 1.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.full(n, 5000.0),
            },
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )
        config = StreakPersistenceConfig()
        result = preprocess(df, config)
        # After 5 consecutive ups (bars 1-5), positive_streak at bar 5 should be 5
        assert result["positive_streak"].iloc[5] == 5
        # After 3 consecutive downs (bars 6-8), negative_streak at bar 8 should be 3
        assert result["negative_streak"].iloc[8] == 3
