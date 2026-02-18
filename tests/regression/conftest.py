"""Regression test fixtures -- 합성 데이터 + BacktestEngine."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.backtest.engine import BacktestEngine
from src.data.market_data import MarketDataSet

_GOLDEN_PATH = Path(__file__).parent / "golden_metrics.yaml"

# 합성 OHLCV 파라미터
_SEED = 42
_N_DAYS = 500
_BASE_PRICE = 50000.0
_ANNUAL_RETURN = 0.5  # 50% annual return (상승 추세)
_DAILY_VOL = 0.03


@pytest.fixture(scope="session")
def golden_metrics() -> dict[str, dict]:
    """Golden metrics YAML 로드."""
    raw = yaml.safe_load(_GOLDEN_PATH.read_text(encoding="utf-8"))
    return raw


@pytest.fixture(scope="session")
def synthetic_ohlcv_btc() -> pd.DataFrame:
    """재현 가능한 합성 BTC/USDT OHLCV (seed=42, 500일).

    GBM (Geometric Brownian Motion) 기반 상승 추세 + 변동성.
    """
    rng = np.random.default_rng(_SEED)

    daily_drift = _ANNUAL_RETURN / 365
    returns = rng.normal(daily_drift, _DAILY_VOL, _N_DAYS)

    prices = _BASE_PRICE * np.cumprod(1 + returns)

    dates = pd.date_range(start="2023-01-01", periods=_N_DAYS, freq="D", tz=UTC)

    df = pd.DataFrame(
        {
            "open": prices * (1 + rng.uniform(-0.005, 0.005, _N_DAYS)),
            "high": prices * (1 + rng.uniform(0.005, 0.025, _N_DAYS)),
            "low": prices * (1 - rng.uniform(0.005, 0.025, _N_DAYS)),
            "close": prices,
            "volume": rng.uniform(1e6, 1e8, _N_DAYS),
        },
        index=dates,
    )
    return df


@pytest.fixture(scope="session")
def btc_market_data(synthetic_ohlcv_btc: pd.DataFrame) -> MarketDataSet:
    """합성 BTC MarketDataSet."""
    idx = synthetic_ohlcv_btc.index
    return MarketDataSet(
        symbol="BTC/USDT",
        timeframe="1D",
        start=idx[0].to_pydatetime(),
        end=idx[-1].to_pydatetime(),
        ohlcv=synthetic_ohlcv_btc,
    )


@pytest.fixture(scope="session")
def backtest_engine() -> BacktestEngine:
    """Session-scoped BacktestEngine."""
    return BacktestEngine()
