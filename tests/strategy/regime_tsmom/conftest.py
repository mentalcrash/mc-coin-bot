"""Regime-TSMOM 테스트 공통 Fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.regime_tsmom.config import RegimeTSMOMConfig


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """D1 sample data for regime-tsmom tests (200 bars)."""
    np.random.seed(42)
    n = 200
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
def trending_ohlcv() -> pd.DataFrame:
    """명확한 상승 추세 데이터 (200 bars)."""
    np.random.seed(123)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    # 강한 상승 추세 (drift 0.5% daily + 작은 noise)
    returns = 0.005 + np.random.randn(n) * 0.002
    close = 50000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.003))
    low = close * (1 - np.abs(np.random.randn(n) * 0.003))
    open_ = close * (1 + np.random.randn(n) * 0.001)
    volume = np.random.uniform(1000, 10000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def volatile_ohlcv() -> pd.DataFrame:
    """급격한 변동 데이터 (200 bars, 방향성 없음)."""
    np.random.seed(456)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    # 큰 랜덤 변동, 방향성 없음
    returns = np.random.choice([-0.03, 0.03], size=n) + np.random.randn(n) * 0.02
    close = 50000 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.uniform(1000, 10000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def default_config() -> RegimeTSMOMConfig:
    return RegimeTSMOMConfig()
