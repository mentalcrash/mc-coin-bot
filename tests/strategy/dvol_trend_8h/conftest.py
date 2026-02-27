"""Shared fixtures for dvol-trend-8h strategy tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Sample OHLCV data for dvol-trend-8h tests."""
    np.random.seed(42)
    n = 300  # Need enough for warmup: max(dvol_percentile_window=252, dc_scale_long=132, vol_window=30) + 10 = 262
    dates = pd.date_range("2024-01-01", periods=n, freq="8h")
    close = 50000 + np.cumsum(np.random.randn(n) * 500)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100
    volume = np.random.uniform(100, 10000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
