"""Shared fixtures for XSMOM strategy tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """D1 sample data for XSMOM tests."""
    np.random.seed(42)
    n = 120  # lookback=21 + vol_window=30 + holding_period 여유분
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
