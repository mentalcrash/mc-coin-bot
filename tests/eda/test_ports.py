"""Port Protocol 준수 테스트.

HistoricalDataFeed, BacktestExecutor, ShadowExecutor가
DataFeedPort/ExecutorPort를 만족하는지 검증합니다.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.data.market_data import MarketDataSet
from src.eda.data_feed import HistoricalDataFeed
from src.eda.executors import BacktestExecutor, ShadowExecutor
from src.eda.ports import DataFeedPort, ExecutorPort
from src.portfolio.cost_model import CostModel


def _make_1m_data(n: int = 60) -> MarketDataSet:
    """최소 1m MarketDataSet 생성."""
    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = pd.date_range(start=start, periods=n, freq="1min", tz=UTC)
    close = 50000.0 + np.cumsum(rng.standard_normal(n) * 1)
    df = pd.DataFrame(
        {
            "open": close * 0.9999,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.integers(10, 100, n) * 10.0,
        },
        index=timestamps,
    )
    return MarketDataSet(
        symbol="BTC/USDT",
        timeframe="1m",
        start=df.index[0].to_pydatetime(),  # type: ignore[union-attr]
        end=df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
        ohlcv=df,
    )


def test_historical_data_feed_satisfies_data_feed_port() -> None:
    """HistoricalDataFeed는 DataFeedPort를 만족한다."""
    feed = HistoricalDataFeed(data=_make_1m_data(), target_timeframe="1h")
    assert isinstance(feed, DataFeedPort)


def test_backtest_executor_satisfies_executor_port() -> None:
    """BacktestExecutor는 ExecutorPort를 만족한다."""
    executor = BacktestExecutor(cost_model=CostModel.zero())
    assert isinstance(executor, ExecutorPort)


def test_shadow_executor_satisfies_executor_port() -> None:
    """ShadowExecutor는 ExecutorPort를 만족한다."""
    executor = ShadowExecutor()
    assert isinstance(executor, ExecutorPort)
