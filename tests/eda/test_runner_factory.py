"""EDA Runner 팩토리 메서드 테스트.

backtest, backtest_agg, shadow 팩토리 및 하위호환 테스트.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.data.market_data import MarketDataSet
from src.eda.data_feed import AggregatingDataFeed, HistoricalDataFeed
from src.eda.executors import BacktestExecutor, ShadowExecutor
from src.eda.runner import EDARunner
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals


class SimpleMovingAverageStrategy(BaseStrategy):
    """테스트용 간단한 이동평균 전략."""

    @property
    def name(self) -> str:
        return "test-sma"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        sma: pd.Series = df["close"].rolling(10).mean()  # type: ignore[assignment]
        df = df.copy()
        df["sma"] = sma
        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        long_signal = df["close"] > df["sma"]
        entries = long_signal & ~long_signal.shift(1, fill_value=False)
        exits = ~long_signal & long_signal.shift(1, fill_value=False)

        direction = pd.Series(0, index=df.index)
        direction[long_signal] = 1

        strength = pd.Series(0.0, index=df.index)
        strength[long_signal] = 1.0

        return StrategySignals(
            entries=entries,
            exits=exits,
            direction=direction.shift(1).fillna(0).astype(int),
            strength=strength.shift(1).fillna(0.0),
        )


def _make_trending_data(n: int = 100, base: float = 50000.0) -> MarketDataSet:
    """상승 트렌드 테스트 데이터."""
    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = pd.date_range(start=start, periods=n, freq="1D", tz=UTC)

    trend = np.linspace(0, 5000, n)
    noise = rng.standard_normal(n) * 200
    close = base + trend + noise

    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100, 1000, n) * 1000.0,
        },
        index=timestamps,
    )
    return MarketDataSet(
        symbol="BTC/USDT",
        timeframe="1D",
        start=timestamps[0].to_pydatetime(),  # type: ignore[union-attr]
        end=timestamps[-1].to_pydatetime(),  # type: ignore[union-attr]
        ohlcv=df,
    )


def _make_config() -> PortfolioManagerConfig:
    return PortfolioManagerConfig(
        max_leverage_cap=2.0,
        rebalance_threshold=0.01,
        cost_model=CostModel.zero(),
    )


class TestRunnerFactoryTypes:
    """팩토리 메서드가 올바른 어댑터 타입을 생성하는지 검증."""

    def test_backtest_factory_creates_correct_types(self) -> None:
        """backtest() → HistoricalDataFeed + BacktestExecutor."""
        data = _make_trending_data(50)
        strategy = SimpleMovingAverageStrategy()
        config = _make_config()

        runner = EDARunner.backtest(strategy=strategy, data=data, config=config)

        assert isinstance(runner._feed, HistoricalDataFeed)
        assert isinstance(runner._executor, BacktestExecutor)

    def test_backtest_agg_factory_creates_correct_types(self) -> None:
        """backtest_agg() → AggregatingDataFeed + BacktestExecutor."""
        data = _make_trending_data(50)
        strategy = SimpleMovingAverageStrategy()
        config = _make_config()

        runner = EDARunner.backtest_agg(
            strategy=strategy,
            data=data,
            target_timeframe="1D",
            config=config,
        )

        assert isinstance(runner._feed, AggregatingDataFeed)
        assert isinstance(runner._executor, BacktestExecutor)

    def test_shadow_factory_creates_correct_types(self) -> None:
        """shadow() → HistoricalDataFeed + ShadowExecutor."""
        data = _make_trending_data(50)
        strategy = SimpleMovingAverageStrategy()
        config = _make_config()

        runner = EDARunner.shadow(strategy=strategy, data=data, config=config)

        assert isinstance(runner._feed, HistoricalDataFeed)
        assert isinstance(runner._executor, ShadowExecutor)

    def test_original_init_backward_compat(self) -> None:
        """기존 __init__ 호출이 여전히 동작."""
        data = _make_trending_data(50)
        strategy = SimpleMovingAverageStrategy()
        config = _make_config()

        runner = EDARunner(strategy=strategy, data=data, config=config)

        assert isinstance(runner._feed, HistoricalDataFeed)
        assert isinstance(runner._executor, BacktestExecutor)


class TestRunnerFactoryRun:
    """팩토리로 생성한 Runner가 실제 run()을 수행하는지 검증."""

    async def test_backtest_factory_run_produces_metrics(self) -> None:
        """backtest() 팩토리 → run() → PerformanceMetrics 반환."""
        data = _make_trending_data(100)
        strategy = SimpleMovingAverageStrategy()
        config = _make_config()

        runner = EDARunner.backtest(
            strategy=strategy, data=data, config=config, initial_capital=10000.0
        )
        metrics = await runner.run()

        assert metrics is not None
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
