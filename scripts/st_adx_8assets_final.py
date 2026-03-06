"""SuperTrend + ADX — 8 Assets 12H 최종 백테스트.

Set B: ATR=7, Mult=2.5, ADX(14, 25), Long-Only, TS 3.0x ATR
에셋: BTC, ETH, SOL, AVAX, XRP, NEAR, UNI, FTM(Sonic)
"""

from __future__ import annotations

import sys
import warnings
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest.engine import BacktestEngine
from src.backtest.request import BacktestRequest
from src.config.settings import get_settings
from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService
from src.portfolio.config import PortfolioManagerConfig
from src.portfolio.portfolio import Portfolio
from src.strategy import get_strategy
from src.strategy.supertrend.config import ShortMode

warnings.filterwarnings("ignore")

# -- Configuration ----------------------------------------------------------
TIMEFRAME = "12h"
START_DATE = datetime(2020, 1, 1, tzinfo=UTC)
END_DATE = datetime(2026, 3, 6, tzinfo=UTC)
INITIAL_CAPITAL = Decimal(100000)

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "AVAX/USDT",
    "XRP/USDT",
    "NEAR/USDT",
    "UNI/USDT",
    "FTM/USDT",
]

PARAMS = {
    "atr_period": 7,
    "multiplier": 2.5,
    "adx_period": 14,
    "adx_threshold": 25,
    "short_mode": ShortMode.DISABLED,
}

PM_CONFIG = PortfolioManagerConfig(
    max_leverage_cap=1.0,
    use_trailing_stop=True,
    trailing_stop_atr_multiplier=3.0,
    rebalance_threshold=0.05,
)


def run_backtest(
    engine: BacktestEngine,
    data_service: MarketDataService,
    strategy: object,
    symbol: str,
) -> dict | None:
    """단일 심볼 백테스트."""
    try:
        config = strategy.config  # type: ignore[attr-defined]
        warmup = config.warmup_periods()

        data = data_service.get(
            MarketDataRequest(
                symbol=symbol,
                timeframe=TIMEFRAME,
                start=START_DATE,
                end=END_DATE,
            )
        )
        if data.periods < warmup + 50:
            logger.warning(f"SKIP {symbol}: insufficient data ({data.periods} bars)")
            return None

        portfolio = Portfolio.create(
            initial_capital=INITIAL_CAPITAL,
            config=PM_CONFIG,
        )
        result = engine.run(
            BacktestRequest(
                data=data,
                strategy=strategy,  # type: ignore[arg-type]
                portfolio=portfolio,
                warmup_bars=warmup,
            )
        )

        m = result.metrics
        bm = result.benchmark
        return {
            "symbol": symbol.replace("/USDT", ""),
            "total_return": round(m.total_return, 2),
            "cagr": round(m.cagr, 2),
            "sharpe": round(m.sharpe_ratio, 2),
            "sortino": round(m.sortino_ratio, 2) if m.sortino_ratio else None,
            "max_drawdown": round(m.max_drawdown, 2),
            "calmar": round(m.cagr / abs(m.max_drawdown) * 100, 2) if m.max_drawdown else None,
            "total_trades": m.total_trades,
            "win_rate": round(m.win_rate, 1),
            "profit_factor": round(m.profit_factor, 2) if m.profit_factor else None,
            "bh_return": round(bm.benchmark_return, 2) if bm else None,
            "beta": round(bm.beta, 2) if bm and bm.beta else None,
        }
    except Exception as e:
        logger.warning(f"SKIP {symbol}: {e}")
        return None


def main() -> None:
    """8 Assets 12H 최종 백테스트."""
    logger.info("=" * 80)
    logger.info("SuperTrend + ADX | 8 Assets 12H | Set B (ATR=7, Mult=2.5)")
    logger.info("Long-Only + TS 3.0x ATR | 2020-01 ~ 2026-03")
    logger.info("=" * 80)

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()

    strategy = get_strategy("supertrend").from_params(**PARAMS)

    results: list[dict] = []
    for symbol in SYMBOLS:
        r = run_backtest(engine, data_service, strategy, symbol)
        if r:
            results.append(r)
            logger.info(
                f"  {r['symbol']:6s} | "
                f"Return {r['total_return']:+8.1f}%  "
                f"CAGR {r['cagr']:+6.1f}%  "
                f"Sharpe {r['sharpe']:5.2f}  "
                f"MDD {r['max_drawdown']:6.1f}%  "
                f"Trades {r['total_trades']:3d}  "
                f"WR {r['win_rate']:4.0f}%  "
                f"PF {r['profit_factor'] or 0:5.2f}  "
                f"B&H {r['bh_return']:+8.1f}%"
            )

    if not results:
        logger.error("No results")
        return

    df = pd.DataFrame(results)

    # -- Full Table --
    print("\n" + "=" * 110)
    print("RESULTS: SuperTrend + ADX (7, 2.5) | 8 Assets 12H | Long-Only + TS 3.0x ATR | 2020-2026")
    print("=" * 110)
    df_sorted = df.sort_values("sharpe", ascending=False)
    print(df_sorted.to_string(index=False))

    # -- Aggregate --
    n = len(df)
    print(f"\n{'=' * 60}")
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    print(f"  Assets:              {n}")
    print(f"  Avg Sharpe:          {df['sharpe'].mean():.3f}")
    print(f"  Median Sharpe:       {df['sharpe'].median():.3f}")
    print(f"  Sharpe >= 1.0:       {(df['sharpe'] >= 1.0).sum()}/{n}")
    print(f"  Sharpe >= 0.5:       {(df['sharpe'] >= 0.5).sum()}/{n}")
    print(f"  Avg Return:          {df['total_return'].mean():+.1f}%")
    print(f"  Avg CAGR:            {df['cagr'].mean():+.1f}%")
    print(f"  Avg MDD:             {df['max_drawdown'].mean():.1f}%")
    print(f"  Avg Win Rate:        {df['win_rate'].mean():.1f}%")
    print(f"  Avg Trades:          {df['total_trades'].mean():.0f}")
    print(f"  Beats B&H:           {(df['total_return'] > df['bh_return']).sum()}/{n}")

    # -- Ranking --
    print("\n--- RANKING by Sharpe ---")
    for i, row in df_sorted.iterrows():
        grade = "★" if row["sharpe"] >= 1.0 else "○" if row["sharpe"] >= 0.5 else "✗"
        print(
            f"  {grade} {row['symbol']:6s}  Sharpe {row['sharpe']:5.2f}  "
            f"CAGR {row['cagr']:+6.1f}%  MDD {row['max_drawdown']:6.1f}%  "
            f"PF {row['profit_factor'] or 0:5.2f}"
        )

    # -- Save --
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "st_adx_8assets_final.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
