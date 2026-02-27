"""Shared fixtures for wavelet-channel-8h strategy tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Sample OHLCV data for wavelet-channel-8h tests."""
    np.random.seed(42)
    n = 200  # Need enough for warmup: scale_long(132) + wavelet_padding(4) + 10 = 146
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
