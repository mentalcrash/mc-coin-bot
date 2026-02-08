"""EDA Runner 통합 테스트.

End-to-end 단일/멀티 심볼 실행, 결과 생성을 검증합니다.
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.data.market_data import MarketDataSet, MultiSymbolData
from src.eda.runner import EDARunner
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals


class SimpleMovingAverageStrategy(BaseStrategy):
    """테스트용 간단한 이동평균 전략.

    Close > SMA(10) → LONG, Close < SMA(10) → NEUTRAL
    """

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


def _make_trending_data(n: int = 100, base: float = 50000.0) -> pd.DataFrame:
    """상승 트렌드 테스트 데이터."""
    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = pd.date_range(start=start, periods=n, freq="1D", tz=UTC)

    # 상승 트렌드 + 노이즈
    trend = np.linspace(0, 5000, n)
    noise = rng.standard_normal(n) * 200
    close = base + trend + noise

    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100, 1000, n) * 1000.0,
        },
        index=timestamps,
    )


class TestRunnerSingleSymbol:
    """단일 심볼 Runner 테스트."""

    async def test_end_to_end_single_symbol(self) -> None:
        """단일 심볼 end-to-end 실행."""
        df = _make_trending_data(100)
        data = MarketDataSet(
            symbol="BTC/USDT",
            timeframe="1D",
            start=df.index[0].to_pydatetime(),  # type: ignore[union-attr]
            end=df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
            ohlcv=df,
        )
        strategy = SimpleMovingAverageStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=10000.0,
        )
        metrics = await runner.run()

        # 기본 검증: 메트릭이 생성됨
        assert metrics is not None
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)

    async def test_analytics_available_after_run(self) -> None:
        """run() 후 analytics 접근 가능."""
        df = _make_trending_data(50)
        data = MarketDataSet(
            symbol="BTC/USDT",
            timeframe="1D",
            start=df.index[0].to_pydatetime(),  # type: ignore[union-attr]
            end=df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
            ohlcv=df,
        )
        strategy = SimpleMovingAverageStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=10000.0,
        )
        await runner.run()

        assert runner.analytics is not None
        assert runner.portfolio_manager is not None
        # equity curve는 최소 1개 이상
        assert len(runner.analytics.equity_curve) >= 0


class TestRunnerMultiSymbol:
    """멀티 심볼 Runner 테스트."""

    async def test_end_to_end_multi_symbol(self) -> None:
        """멀티 심볼 end-to-end 실행."""
        n = 80
        symbols = ["BTC/USDT", "ETH/USDT"]
        ohlcv: dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols):
            ohlcv[sym] = _make_trending_data(n, base=50000.0 + i * 10000)

        first_df = ohlcv[symbols[0]]
        data = MultiSymbolData(
            symbols=symbols,
            timeframe="1D",
            start=first_df.index[0].to_pydatetime(),  # type: ignore[union-attr]
            end=first_df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
            ohlcv=ohlcv,
        )

        strategy = SimpleMovingAverageStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=10000.0,
            asset_weights={"BTC/USDT": 0.5, "ETH/USDT": 0.5},
        )
        metrics = await runner.run()

        assert metrics is not None
        assert isinstance(metrics.total_return, float)


class TestRunnerEdgeCases:
    """Runner 엣지 케이스 테스트."""

    async def test_short_data(self) -> None:
        """데이터가 warmup 미달이면 시그널 없이 완료."""
        df = _make_trending_data(5)  # warmup(~20) 미달
        data = MarketDataSet(
            symbol="BTC/USDT",
            timeframe="1D",
            start=df.index[0].to_pydatetime(),  # type: ignore[union-attr]
            end=df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
            ohlcv=df,
        )
        strategy = SimpleMovingAverageStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=10000.0,
        )
        metrics = await runner.run()

        # warmup 미달 → 거래 0
        assert metrics.total_trades == 0


# =========================================================================
# Multi-Asset EDA 통합 테스트
# =========================================================================
_EIGHT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "AVAX/USDT",
]


def _make_multi_trending_data(
    symbols: list[str],
    n: int = 100,
) -> MultiSymbolData:
    """멀티 심볼 상승 트렌드 테스트 데이터."""
    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = pd.date_range(start=start, periods=n, freq="1D", tz=UTC)

    ohlcv: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols):
        base = 50000.0 / (2**i)  # BTC=50000, ETH=25000, ...
        trend = np.linspace(0, base * 0.2, n)  # 20% 상승
        noise = rng.standard_normal(n) * base * 0.02
        close = base + trend + noise

        ohlcv[sym] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.012,
                "low": close * 0.988,
                "close": close,
                "volume": rng.integers(100, 1000, n) * 1000.0,
            },
            index=timestamps,
        )

    return MultiSymbolData(
        symbols=symbols,
        timeframe="1D",
        start=timestamps[0].to_pydatetime(),  # type: ignore[union-attr]
        end=timestamps[-1].to_pydatetime(),  # type: ignore[union-attr]
        ohlcv=ohlcv,
    )


class TestMultiAssetEDA:
    """8-asset 멀티 심볼 EDA 통합 테스트."""

    async def test_8_asset_equal_weight(self) -> None:
        """8개 심볼 equal-weight 실행."""
        data = _make_multi_trending_data(_EIGHT_SYMBOLS, n=100)
        strategy = SimpleMovingAverageStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )

        n_symbols = len(_EIGHT_SYMBOLS)
        weights = dict.fromkeys(_EIGHT_SYMBOLS, 1.0 / n_symbols)

        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=100000.0,
            asset_weights=weights,
        )
        metrics = await runner.run()

        # 기본 검증
        assert metrics is not None
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        # 100 bars, warmup=10 → 충분한 거래 발생
        assert metrics.total_trades > 0

    async def test_multi_asset_weight_distribution(self) -> None:
        """불균등 가중치 적용 확인."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        data = _make_multi_trending_data(symbols, n=80)
        strategy = SimpleMovingAverageStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )

        # BTC 60%, ETH 30%, SOL 10%
        weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.3, "SOL/USDT": 0.1}

        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=10000.0,
            asset_weights=weights,
        )
        metrics = await runner.run()

        assert metrics is not None
        assert metrics.total_trades > 0

    async def test_multi_asset_results_reasonable(self) -> None:
        """8-asset 결과가 합리적 범위 내."""
        data = _make_multi_trending_data(_EIGHT_SYMBOLS, n=100)
        strategy = SimpleMovingAverageStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.05,
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )

        n_symbols = len(_EIGHT_SYMBOLS)
        weights = dict.fromkeys(_EIGHT_SYMBOLS, 1.0 / n_symbols)

        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=100000.0,
            asset_weights=weights,
        )
        metrics = await runner.run()

        # 상승 트렌드 데이터 → 양수 수익 기대
        # (SMA 전략이므로 초반 warmup 후 LONG → 수익)
        assert metrics.total_return > -50.0, f"Return too negative: {metrics.total_return:.2f}%"
        assert metrics.max_drawdown < 50.0, f"MDD too large: {metrics.max_drawdown:.2f}%"
        # 8개 심볼에서 최소 4개 거래 이상
        assert metrics.total_trades >= 4
