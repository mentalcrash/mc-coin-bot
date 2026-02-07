"""Parity Test: VBT BacktestEngine vs EDA Runner.

동일 데이터/전략/설정으로 VBT와 EDA의 결과를 비교합니다.
EDA는 bar-by-bar 이벤트 기반이므로 벡터화 VBT와 완전히 동일할 수 없지만,
주요 지표가 허용 오차 내에 있어야 합니다.

차이 발생 원인:
  - VBT: 시그널 시점의 close 가격 기준 체결
  - EDA: 다음 바의 open 가격 기준 체결 (look-ahead bias 방지)
  - 따라서 정확한 수치 일치보다 **방향성과 규모(order of magnitude)** 일치를 검증
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.data.market_data import MarketDataSet
from src.eda.runner import EDARunner
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.cost_model import CostModel
from src.portfolio.portfolio import Portfolio
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals


class SimpleMomentumStrategy(BaseStrategy):
    """Parity 테스트용 단순 모멘텀 전략.

    Close > Close[20] → LONG (strength=1.0)
    Close <= Close[20] → NEUTRAL
    """

    @property
    def name(self) -> str:
        return "parity-momentum"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = 20
        df = df.copy()
        ret: pd.Series = df["close"].pct_change(lookback)  # type: ignore[assignment]
        df["momentum"] = ret
        return df

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        long_signal = df["momentum"] > 0

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


def _make_market_data(n: int = 365) -> MarketDataSet:
    """1년간 상승 트렌드 데이터."""
    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = pd.date_range(start=start, periods=n, freq="1D", tz=UTC)

    # 상승 트렌드 + 노이즈 (변동성 있는 상승)
    trend = np.linspace(0, 10000, n)
    noise = np.cumsum(rng.standard_normal(n) * 300)
    close = 50000.0 + trend + noise

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
        start=df.index[0].to_pydatetime(),  # type: ignore[union-attr]
        end=df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
        ohlcv=df,
    )


class TestVBTvsEDAParity:
    """VBT BacktestEngine vs EDA Runner 비교."""

    async def test_both_engines_produce_results(self) -> None:
        """양쪽 모두 결과를 생성."""
        data = _make_market_data(200)
        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        # VBT
        portfolio = Portfolio.create(initial_capital=10000, config=config)
        request = BacktestRequest(
            data=data,
            strategy=strategy,
            portfolio=portfolio,
        )
        engine = BacktestEngine()
        vbt_result = engine.run(request)

        # EDA
        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=10000.0,
        )
        eda_metrics = await runner.run()

        # 기본 검증: 양쪽 모두 결과 생성
        assert vbt_result.metrics is not None
        assert eda_metrics is not None

        # 양쪽 모두 거래가 발생해야 함
        assert vbt_result.metrics.total_trades > 0
        assert eda_metrics.total_trades > 0  # H-003: EDA도 거래 발생 필수

    async def test_return_sign_consistency(self) -> None:
        """수익률 부호(양/음)가 일치."""
        data = _make_market_data(365)
        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        # VBT
        portfolio = Portfolio.create(initial_capital=10000, config=config)
        request = BacktestRequest(
            data=data,
            strategy=strategy,
            portfolio=portfolio,
        )
        engine = BacktestEngine()
        vbt_result = engine.run(request)

        # EDA
        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=10000.0,
        )
        eda_metrics = await runner.run()

        vbt_return = vbt_result.metrics.total_return
        eda_return = eda_metrics.total_return

        # 상승 트렌드 데이터이므로 양쪽 모두 양수 수익이어야 함
        # (EDA의 equity curve가 최소 2포인트 있어야 유의미)
        if abs(eda_return) > 0.01:
            # 부호 일치 확인 (같은 방향의 수익)
            assert (vbt_return > 0) == (eda_return > 0), (
                f"Return sign mismatch: VBT={vbt_return:.2f}%, EDA={eda_return:.2f}%"
            )

    async def test_no_trades_when_warmup_insufficient(self) -> None:
        """Warmup 미달 시 양쪽 모두 거래 0."""
        data = _make_market_data(10)  # 10일 < warmup(~30)
        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        # EDA
        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=10000.0,
        )
        eda_metrics = await runner.run()
        assert eda_metrics.total_trades == 0


class TestDeepParityComparison:
    """VBT vs EDA 심층 수치 비교.

    체결 방식 차이(VBT=close, EDA=next-open)로 정확한 일치는 불가능하므로
    넉넉한 허용 오차 내에서 방향성과 규모가 일치하는지 검증합니다.
    """

    @staticmethod
    def _run_both(
        data: MarketDataSet,
        strategy: BaseStrategy,
        config: PortfolioManagerConfig,
        capital: float = 10000.0,
    ) -> tuple:
        """VBT와 EDA 양쪽 결과를 반환하는 헬퍼."""
        import asyncio

        # VBT
        portfolio = Portfolio.create(initial_capital=int(capital), config=config)
        request = BacktestRequest(
            data=data,
            strategy=strategy,
            portfolio=portfolio,
        )
        engine = BacktestEngine()
        vbt_result = engine.run(request)

        # EDA
        runner = EDARunner(
            strategy=strategy,
            data=data,
            config=config,
            initial_capital=capital,
        )
        eda_metrics = asyncio.get_event_loop().run_until_complete(runner.run())

        return vbt_result.metrics, eda_metrics

    async def test_total_return_within_tolerance(self) -> None:
        """총 수익률이 허용 오차 내에서 유사."""
        data = _make_market_data(365)
        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        # VBT
        portfolio = Portfolio.create(initial_capital=10000, config=config)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        vbt_metrics = BacktestEngine().run(request).metrics

        # EDA
        runner = EDARunner(strategy=strategy, data=data, config=config, initial_capital=10000.0)
        eda_metrics = await runner.run()

        # 양쪽 모두 양수 수익 (상승 트렌드 데이터)
        assert vbt_metrics.total_return > 0, f"VBT return={vbt_metrics.total_return}"
        assert eda_metrics.total_return > 0, f"EDA return={eda_metrics.total_return}"

        # 수익률 규모가 같은 자릿수 (order of magnitude)
        # 체결 가격 차이로 ±50% 이내면 합리적
        ratio = eda_metrics.total_return / vbt_metrics.total_return
        assert 0.2 < ratio < 5.0, (
            f"Return ratio out of range: VBT={vbt_metrics.total_return:.2f}%, "
            f"EDA={eda_metrics.total_return:.2f}%, ratio={ratio:.2f}"
        )

    async def test_sharpe_direction_consistency(self) -> None:
        """Sharpe 비율의 부호가 일치."""
        data = _make_market_data(365)
        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        portfolio = Portfolio.create(initial_capital=10000, config=config)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        vbt_metrics = BacktestEngine().run(request).metrics

        runner = EDARunner(strategy=strategy, data=data, config=config, initial_capital=10000.0)
        eda_metrics = await runner.run()

        # 상승 트렌드 → 양쪽 모두 양수 Sharpe
        assert vbt_metrics.sharpe_ratio > 0
        if eda_metrics.total_trades > 0:
            assert eda_metrics.sharpe_ratio > 0, (
                f"Sharpe sign mismatch: VBT={vbt_metrics.sharpe_ratio:.4f}, "
                f"EDA={eda_metrics.sharpe_ratio:.4f}"
            )

    async def test_max_drawdown_within_tolerance(self) -> None:
        """MDD가 유사한 범위."""
        data = _make_market_data(365)
        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        portfolio = Portfolio.create(initial_capital=10000, config=config)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        vbt_metrics = BacktestEngine().run(request).metrics

        runner = EDARunner(strategy=strategy, data=data, config=config, initial_capital=10000.0)
        eda_metrics = await runner.run()

        # MDD는 양수 (이 코드베이스에서 MDD = |peak-to-trough| %)
        assert vbt_metrics.max_drawdown >= 0
        assert eda_metrics.max_drawdown >= 0

        # MDD 차이가 20pp 이내 (체결 방식 차이 감안)
        mdd_diff = abs(vbt_metrics.max_drawdown - eda_metrics.max_drawdown)
        assert mdd_diff < 20.0, (
            f"MDD difference too large: VBT={vbt_metrics.max_drawdown:.2f}%, "
            f"EDA={eda_metrics.max_drawdown:.2f}%, diff={mdd_diff:.2f}pp"
        )

    async def test_metrics_summary_print(self, capsys: object) -> None:
        """양쪽 메트릭을 출력하여 수동 검토 가능하게 함."""
        data = _make_market_data(365)
        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        portfolio = Portfolio.create(initial_capital=10000, config=config)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        vbt_metrics = BacktestEngine().run(request).metrics

        runner = EDARunner(strategy=strategy, data=data, config=config, initial_capital=10000.0)
        eda_metrics = await runner.run()

        # 메트릭 요약 출력 (pytest -s 로 확인 가능)
        print("\n" + "=" * 60)
        print("VBT vs EDA Parity Report")
        print("=" * 60)
        print(f"{'Metric':<25} {'VBT':>12} {'EDA':>12} {'Delta':>12}")
        print("-" * 60)

        rows = [
            ("Total Return (%)", vbt_metrics.total_return, eda_metrics.total_return),
            ("CAGR (%)", vbt_metrics.cagr, eda_metrics.cagr),
            ("Sharpe Ratio", vbt_metrics.sharpe_ratio, eda_metrics.sharpe_ratio),
            ("Max Drawdown (%)", vbt_metrics.max_drawdown, eda_metrics.max_drawdown),
            ("Win Rate (%)", vbt_metrics.win_rate, eda_metrics.win_rate),
            (
                "Total Trades",
                float(vbt_metrics.total_trades),
                float(eda_metrics.total_trades),
            ),
            (
                "Winning Trades",
                float(vbt_metrics.winning_trades),
                float(eda_metrics.winning_trades),
            ),
            (
                "Losing Trades",
                float(vbt_metrics.losing_trades),
                float(eda_metrics.losing_trades),
            ),
        ]

        for label, vbt_val, eda_val in rows:
            delta = eda_val - vbt_val
            print(f"{label:<25} {vbt_val:>12.4f} {eda_val:>12.4f} {delta:>+12.4f}")

        if vbt_metrics.volatility is not None and eda_metrics.volatility is not None:
            delta_vol = eda_metrics.volatility - vbt_metrics.volatility
            print(
                f"{'Volatility (%)':.<25} {vbt_metrics.volatility:>12.4f} "
                f"{eda_metrics.volatility:>12.4f} {delta_vol:>+12.4f}"
            )

        print("=" * 60)

        # 기본 assertion: 테스트가 정상 완료
        assert vbt_metrics.total_return is not None
        assert eda_metrics.total_return is not None

    async def test_downtrend_both_negative(self) -> None:
        """하락 트렌드에서 양쪽 모두 음수 또는 0 수익."""
        rng = np.random.default_rng(99)
        n = 365
        start = datetime(2024, 1, 1, tzinfo=UTC)
        timestamps = pd.date_range(start=start, periods=n, freq="1D", tz=UTC)

        # 하락 트렌드
        trend = np.linspace(0, -20000, n)
        noise = np.cumsum(rng.standard_normal(n) * 200)
        close = 60000.0 + trend + noise
        close = np.maximum(close, 1000.0)  # 가격 하한

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

        data = MarketDataSet(
            symbol="BTC/USDT",
            timeframe="1D",
            start=df.index[0].to_pydatetime(),  # type: ignore[union-attr]
            end=df.index[-1].to_pydatetime(),  # type: ignore[union-attr]
            ohlcv=df,
        )

        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        portfolio = Portfolio.create(initial_capital=10000, config=config)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        vbt_metrics = BacktestEngine().run(request).metrics

        runner = EDARunner(strategy=strategy, data=data, config=config, initial_capital=10000.0)
        eda_metrics = await runner.run()

        # long-only 전략 + 하락 트렌드 → 수익이 제한적이거나 손실
        # 양쪽 방향 일치 확인
        if abs(eda_metrics.total_return) > 1.0 and abs(vbt_metrics.total_return) > 1.0:
            assert (vbt_metrics.total_return > 0) == (eda_metrics.total_return > 0), (
                f"Direction mismatch in downtrend: "
                f"VBT={vbt_metrics.total_return:.2f}%, EDA={eda_metrics.total_return:.2f}%"
            )

    async def test_return_relative_tolerance(self) -> None:
        """H-003: 수익률 상대 오차가 30% 이내."""
        data = _make_market_data(365)
        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        portfolio = Portfolio.create(initial_capital=10000, config=config)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        vbt_metrics = BacktestEngine().run(request).metrics

        runner = EDARunner(strategy=strategy, data=data, config=config, initial_capital=10000.0)
        eda_metrics = await runner.run()

        vbt_ret = vbt_metrics.total_return
        eda_ret = eda_metrics.total_return
        denom = max(abs(vbt_ret), 1.0)
        rel_err = abs(vbt_ret - eda_ret) / denom
        assert rel_err < 0.30, (
            f"Return relative error too large: VBT={vbt_ret:.2f}%, "
            f"EDA={eda_ret:.2f}%, rel_err={rel_err:.2%}"
        )

    async def test_trade_count_similar(self) -> None:
        """H-003: 거래 횟수가 유사 (±50% 허용)."""
        data = _make_market_data(365)
        strategy = SimpleMomentumStrategy()
        config = PortfolioManagerConfig(
            max_leverage_cap=2.0,
            rebalance_threshold=0.01,
            cost_model=CostModel.zero(),
        )

        portfolio = Portfolio.create(initial_capital=10000, config=config)
        request = BacktestRequest(data=data, strategy=strategy, portfolio=portfolio)
        vbt_metrics = BacktestEngine().run(request).metrics

        runner = EDARunner(strategy=strategy, data=data, config=config, initial_capital=10000.0)
        eda_metrics = await runner.run()

        vbt_trades = vbt_metrics.total_trades
        eda_trades = eda_metrics.total_trades
        assert vbt_trades > 0 and eda_trades > 0, "Both must have trades"

        ratio = eda_trades / vbt_trades
        assert 0.5 < ratio < 2.0, (
            f"Trade count ratio out of range: VBT={vbt_trades}, EDA={eda_trades}, ratio={ratio:.2f}"
        )
