"""Shared fixtures for tests.

이 모듈은 테스트에서 공통으로 사용되는 픽스처를 제공합니다.

Rules Applied:
    - #17 Testing Standards: Pytest fixtures
    - #12 Data Engineering: Sample data generation
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 디렉토리 경로 → pytest 마커 자동 매핑
# ---------------------------------------------------------------------------
_DIR_MARKER_MAP: dict[str, str] = {
    "/strategy/": "strategy",
    "/eda/": "eda",
    "/chaos/": "chaos",
    "/data/": "data",
    "/backtest/": "integration",
    "/regression/": "slow",
    "/orchestrator/": "integration",
    "/notification/": "integration",
    "/exchange/": "integration",
    "/pipeline/": "integration",
    "/cli/": "integration",
    "/core/": "unit",
    "/models/": "unit",
    "/config/": "unit",
    "/market/": "unit",
    "/regime/": "unit",
    "/monitoring/": "unit",
    "/catalog/": "unit",
    "/portfolio/": "unit",
    "/logging/": "unit",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """디렉토리 경로 기반 자동 마커 부여."""
    for item in items:
        fspath = str(item.fspath)
        for dir_pattern, marker_name in _DIR_MARKER_MAP.items():
            if dir_pattern in fspath:
                item.add_marker(getattr(pytest.mark, marker_name))
                break

# Opt-in to future pandas behavior: fillna/ffill/bfill won't auto-downcast
pd.set_option("future.no_silent_downcasting", True)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """샘플 OHLCV DataFrame을 생성합니다.

    100일간의 가상 가격 데이터를 생성합니다.
    가격은 상승 추세 + 랜덤 노이즈로 구성됩니다.
    """
    np.random.seed(42)
    n_periods = 100

    # 시작 시간: 100일 전
    start_time = datetime.now(UTC) - timedelta(days=n_periods)
    timestamps = pd.date_range(start=start_time, periods=n_periods, freq="1D", tz=UTC)

    # 기본 가격: 상승 추세 + 랜덤 노이즈
    base_price = 50000.0
    trend = np.linspace(0, 5000, n_periods)  # 5000 상승
    noise = np.random.randn(n_periods) * 500  # 500 노이즈
    close_prices = base_price + trend + noise

    # OHLCV 생성
    high_prices = close_prices * (1 + np.random.rand(n_periods) * 0.02)
    low_prices = close_prices * (1 - np.random.rand(n_periods) * 0.02)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price
    volume = np.random.randint(100, 1000, n_periods) * 1000

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        },
        index=timestamps,
    )

    return df


@pytest.fixture
def sample_processed_df(sample_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """전처리된 DataFrame을 생성합니다."""
    from src.strategy.tsmom.config import TSMOMConfig
    from src.strategy.tsmom.preprocessor import preprocess

    config = TSMOMConfig(
        lookback=20,
        vol_window=20,
        vol_target=0.40,
    )

    return preprocess(sample_ohlcv_df, config)


@pytest.fixture
def sample_benchmark_returns(sample_ohlcv_df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
    """벤치마크 수익률 시리즈를 생성합니다."""
    close: pd.Series = sample_ohlcv_df["close"]  # type: ignore[assignment]
    returns = close.pct_change().dropna()
    returns.name = "benchmark"
    return returns


@pytest.fixture
def sample_diagnostics_df(sample_processed_df: pd.DataFrame) -> pd.DataFrame:
    """샘플 진단 DataFrame을 생성합니다."""
    from src.strategy.tsmom.config import TSMOMConfig
    from src.strategy.tsmom.signal import generate_signals_with_diagnostics

    config = TSMOMConfig(
        lookback=20,
        vol_window=20,
        vol_target=0.40,
    )

    result = generate_signals_with_diagnostics(sample_processed_df, config, symbol="BTC/USDT")
    df = result.diagnostics_df

    # beta_attribution 테스트에서 필요한 컬럼 보충 (Pure TSMOM에서는 deadband 미사용)
    if "signal_after_deadband" not in df.columns:
        df["signal_after_deadband"] = df["scaled_momentum"]

    return df
