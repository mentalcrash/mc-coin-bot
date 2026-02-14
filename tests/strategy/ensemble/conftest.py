"""Shared fixtures for ensemble strategy tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """D1 sample data for ensemble strategy tests."""
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100
    volume = np.random.uniform(100, 10000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def agg_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Direction/strength matrices + weights for aggregator tests.

    Returns:
        (directions, strengths, weights) 3개 전략, 100 bars
    """
    np.random.seed(123)
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="1D")

    directions = pd.DataFrame(
        {
            "s0": np.random.choice([-1, 0, 1], size=n),
            "s1": np.random.choice([-1, 0, 1], size=n),
            "s2": np.random.choice([-1, 0, 1], size=n),
        },
        index=idx,
    )
    strengths = pd.DataFrame(
        {
            "s0": np.random.randn(n) * 0.5,
            "s1": np.random.randn(n) * 0.3,
            "s2": np.random.randn(n) * 0.4,
        },
        index=idx,
    )
    weights = pd.Series({"s0": 1.0, "s1": 1.0, "s2": 1.0})

    return directions, strengths, weights
