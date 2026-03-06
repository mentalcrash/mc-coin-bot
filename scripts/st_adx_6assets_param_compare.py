"""SuperTrend + ADX — 6 Assets 12H 파라미터 비교.

Set A: ATR=10, Mult=3.0 (기존)
Set B: ATR=7,  Mult=2.5 (민감)
공통: ADX(14, 25), Long-Only, TS 3.0x ATR
에셋: BTC, ETH, SOL, AVAX, XRP, NEAR
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
]

PARAM_SETS = {
    "A (10, 3.0)": {
        "atr_period": 10,
        "multiplier": 3.0,
        "adx_period": 14,
        "adx_threshold": 25,
        "short_mode": ShortMode.DISABLED,
    },
    "B (7, 2.5)": {
        "atr_period": 7,
        "multiplier": 2.5,
        "adx_period": 14,
        "adx_threshold": 25,
        "short_mode": ShortMode.DISABLED,
    },
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
    param_name: str,
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
            "params": param_name,
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
        }
    except Exception as e:
        logger.warning(f"SKIP {symbol} {param_name}: {e}")
        return None


def main() -> None:
    """6 Assets 파라미터 비교 백테스트."""
    logger.info("=" * 80)
    logger.info("SuperTrend + ADX | 6 Assets 12H | Parameter Comparison")
    logger.info("Set A: ATR=10, Mult=3.0  vs  Set B: ATR=7, Mult=2.5")
    logger.info("=" * 80)

    settings = get_settings()
    data_service = MarketDataService(settings)
    engine = BacktestEngine()

    all_results: list[dict] = []

    for param_name, params in PARAM_SETS.items():
        strategy = get_strategy("supertrend").from_params(**params)
        logger.info(f"\n{'=' * 40} {param_name} {'=' * 40}")

        for symbol in SYMBOLS:
            r = run_backtest(engine, data_service, strategy, symbol, param_name)
            if r:
                all_results.append(r)
                logger.info(
                    f"  {r['symbol']:6s} | "
                    f"Return {r['total_return']:+8.1f}%  "
                    f"Sharpe {r['sharpe']:5.2f}  "
                    f"MDD {r['max_drawdown']:6.1f}%  "
                    f"Trades {r['total_trades']:3d}  "
                    f"WR {r['win_rate']:4.0f}%  "
                    f"PF {r['profit_factor'] or 0:5.2f}"
                )

    if not all_results:
        logger.error("No results")
        return

    df = pd.DataFrame(all_results)

    # -- Per-asset comparison --
    print("\n" + "=" * 100)
    print("FULL RESULTS")
    print("=" * 100)
    print(df.to_string(index=False))

    # -- Side-by-side comparison --
    df_a = df[df["params"].str.startswith("A")].set_index("symbol")
    df_b = df[df["params"].str.startswith("B")].set_index("symbol")

    compare = pd.DataFrame(
        {
            "A_sharpe": df_a["sharpe"],
            "B_sharpe": df_b["sharpe"],
            "d_sharpe": df_b["sharpe"] - df_a["sharpe"],
            "A_return": df_a["total_return"],
            "B_return": df_b["total_return"],
            "A_mdd": df_a["max_drawdown"],
            "B_mdd": df_b["max_drawdown"],
            "d_mdd": df_b["max_drawdown"] - df_a["max_drawdown"],
            "A_trades": df_a["total_trades"],
            "B_trades": df_b["total_trades"],
            "A_wr": df_a["win_rate"],
            "B_wr": df_b["win_rate"],
            "A_pf": df_a["profit_factor"],
            "B_pf": df_b["profit_factor"],
        }
    )

    print(f"\n{'=' * 100}")
    print("SIDE-BY-SIDE: A (10, 3.0) vs B (7, 2.5)")
    print("=" * 100)
    print(compare.to_string())

    # -- Aggregate --
    print(f"\n{'=' * 60}")
    print("AGGREGATE")
    print("=" * 60)
    for label, sub in [("A (10, 3.0)", df_a), ("B (7, 2.5)", df_b)]:
        avg_sh = sub["sharpe"].mean()
        med_sh = sub["sharpe"].median()
        avg_ret = sub["total_return"].mean()
        avg_mdd = sub["max_drawdown"].mean()
        avg_wr = sub["win_rate"].mean()
        avg_trades = sub["total_trades"].mean()
        sh_ge1 = (sub["sharpe"] >= 1.0).sum()
        print(f"\n  {label}:")
        print(f"    Avg Sharpe:    {avg_sh:.3f}  (Median: {med_sh:.3f})")
        print(f"    Avg Return:    {avg_ret:+.1f}%")
        print(f"    Avg MDD:       {avg_mdd:.1f}%")
        print(f"    Avg WinRate:   {avg_wr:.1f}%")
        print(f"    Avg Trades:    {avg_trades:.0f}")
        print(f"    Sharpe >= 1.0: {sh_ge1}/6")

    # Sharpe 개선/악화
    improved = (compare["d_sharpe"] > 0).sum()
    degraded = (compare["d_sharpe"] < 0).sum()
    print(f"\n  B vs A: Sharpe improved {improved}/6, degraded {degraded}/6")
    print(f"  Avg Sharpe delta: {compare['d_sharpe'].mean():+.3f}")

    # -- Save --
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "st_adx_6assets_param_compare.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
